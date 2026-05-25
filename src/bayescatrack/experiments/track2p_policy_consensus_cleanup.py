"""Consensus bridge-split cleanup for Track2p-policy experiments.

The Track2p-policy reproduction is intentionally close to Track2p, but recent
error audits showed that the most useful accuracy lever is conservative removal
of false-positive continuations. This module combines two independent signals
before cutting a bridge:

* component-level edge risk from the weakest-bridge cleanup diagnostics, and
* threshold-stability support across nearby IoU-distance thresholds.

The default mode splits only bridges that are both high-risk and unstable. This
keeps the operation prune-only and conservative while making it runnable through
the normal benchmark CLI.
"""

from __future__ import annotations

# jscpd:ignore-start
import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _filter_tracks_by_seed_rois,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _reference_seed_roi_set,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentAuditOutput,
    ComponentCleanupConfig,
    _diagnostics_by_suite2p_edge,
    _no_prune_config,
    _normalize_int_track_matrix,
    _valid_seed_roi,
    component_audit_rows,
    edge_risk_score,
    write_component_rows,
)
from bayescatrack.experiments.track2p_policy_multisplit_cleanup import (
    _select_optimal_split_indices,
    apply_ranked_bridge_splits,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)
from bayescatrack.experiments.track2p_policy_stability_cleanup import (
    Edge,
    StabilityCleanupConfig,
    edge_support_counts,
)

TRACK2P_POLICY_CONSENSUS_CLEANUP_METHOD = "track2p-policy-consensus-cleanup"

ConsensusMode = Literal[
    "risk-and-stability",
    "risk-or-stability",
    "risk-only",
    "stability-only",
]
CONSENSUS_MODES = (
    "risk-and-stability",
    "risk-or-stability",
    "risk-only",
    "stability-only",
)


@dataclass(frozen=True)
class ConsensusSplitConfig:
    """Controls how edge-risk and threshold-stability evidence are combined."""

    component: ComponentCleanupConfig = field(default_factory=ComponentCleanupConfig)
    required_support_votes: int = 2
    max_splits_per_component: int = 2
    mode: ConsensusMode = "risk-and-stability"

    def __post_init__(self) -> None:
        if int(self.required_support_votes) < 1:
            raise ValueError("required_support_votes must be at least 1")
        if int(self.max_splits_per_component) < 1:
            raise ValueError("max_splits_per_component must be at least 1")
        if self.mode not in CONSENSUS_MODES:
            raise ValueError("unsupported consensus mode")


@dataclass(frozen=True)
class ConsensusCleanupConfig:
    """Benchmark-level configuration for consensus Track2p-policy cleanup."""

    component: ComponentCleanupConfig = field(default_factory=ComponentCleanupConfig)
    stability: StabilityCleanupConfig = field(default_factory=StabilityCleanupConfig)
    max_splits_per_component: int = 2
    mode: ConsensusMode = "risk-and-stability"

    def __post_init__(self) -> None:
        if int(self.max_splits_per_component) < 1:
            raise ValueError("max_splits_per_component must be at least 1")
        if self.mode not in CONSENSUS_MODES:
            raise ValueError("unsupported consensus mode")

    @property
    def split_config(self) -> ConsensusSplitConfig:
        """Return the helper config implied by the benchmark settings."""

        return ConsensusSplitConfig(
            component=self.component,
            required_support_votes=self.stability.required_support_votes,
            max_splits_per_component=self.max_splits_per_component,
            mode=self.mode,
        )


def run_track2p_policy_consensus_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ConsensusCleanupConfig | None = None,
    apply_splits: bool = True,
) -> ComponentAuditOutput:
    """Run Track2p-policy and split bridges with consensus risk evidence."""

    cleanup = cleanup_config or ConsensusCleanupConfig()
    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    component_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy consensus cleanup requires independent manual "
                "GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)

        base_prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=cleanup.stability.base_iou_distance_threshold,
            prune_config=_no_prune_config(),
        )
        ensemble_predictions = tuple(
            emulate_track2p_pruned_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=threshold,
                prune_config=_no_prune_config(),
            ).tracks
            for threshold in cleanup.stability.ensemble_iou_distance_thresholds
        )
        support_counts = edge_support_counts(ensemble_predictions)
        diagnostics_by_edge = _diagnostics_by_suite2p_edge(
            sessions, base_prediction.diagnostics
        )
        predicted_full = _normalize_int_track_matrix(base_prediction.tracks)
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        predicted_eval, reference_eval, evaluated_track_ids = (
            _evaluated_prediction_rows(
                predicted_full,
                reference_tracks,
                config=policy_config,
            )
        )
        subject_rows = component_audit_rows(
            predicted_eval,
            reference_eval,
            sessions=sessions,
            diagnostics=base_prediction.diagnostics,
            subject=subject_dir.name,
            config=cleanup.component,
            track_ids=evaluated_track_ids,
            seed_session=policy_config.seed_session,
        )
        split_plan = plan_consensus_bridge_splits(
            predicted_eval,
            diagnostics_by_edge=diagnostics_by_edge,
            support_counts=support_counts,
            config=cleanup.split_config,
            track_ids=evaluated_track_ids,
        )
        enriched_rows = _enrich_component_rows_with_split_plan(
            subject_rows,
            split_plan,
            support_counts=support_counts,
            apply_splits=apply_splits,
        )
        cleaned = (
            apply_consensus_bridge_splits(predicted_full, split_plan)
            if apply_splits
            else predicted_full
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        candidate_splits = int(sum(len(splits) for splits in split_plan.values()))
        applied_splits = int(candidate_splits if apply_splits else 0)
        scores = {
            **scores,
            "track2p_policy_variant": TRACK2P_POLICY_CONSENSUS_CLEANUP_METHOD,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_base_iou_distance_threshold": float(
                cleanup.stability.base_iou_distance_threshold
            ),
            "track2p_policy_stability_iou_distance_thresholds": json.dumps(
                list(cleanup.stability.ensemble_iou_distance_thresholds)
            ),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_policy_consensus_mode": str(cleanup.mode),
            "track2p_policy_consensus_required_support_votes": int(
                cleanup.stability.required_support_votes
            ),
            "track2p_policy_consensus_max_splits_per_component": int(
                cleanup.max_splits_per_component
            ),
            "track2p_component_apply_splits": int(apply_splits),
            "track2p_component_candidate_splits": candidate_splits,
            "track2p_component_applied_splits": applied_splits,
            "track2p_component_split_risk_threshold": float(
                cleanup.component.split_risk_threshold
            ),
            "track2p_component_split_penalty": float(cleanup.component.split_penalty),
            "track2p_component_min_side_observations": int(
                cleanup.component.min_side_observations
            ),
            "track2p_component_require_complete_track": int(
                cleanup.component.require_complete_track
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=(
                    "Track2p-policy consensus bridge cleanup"
                    if apply_splits
                    else "Track2p-policy consensus bridge audit"
                ),
                method=cast(Any, TRACK2P_POLICY_CONSENSUS_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        component_rows.extend(
            _with_metadata(
                enriched_rows,
                {
                    "threshold_method": str(threshold_method),
                    "base_iou_distance_threshold": float(
                        cleanup.stability.base_iou_distance_threshold
                    ),
                    "stability_iou_distance_thresholds": json.dumps(
                        list(cleanup.stability.ensemble_iou_distance_thresholds)
                    ),
                    "required_support_votes": int(
                        cleanup.stability.required_support_votes
                    ),
                    "cell_probability_threshold": float(
                        policy_config.cell_probability_threshold
                    ),
                    "transform_type": str(policy_config.transform_type),
                    "cleanup_method": TRACK2P_POLICY_CONSENSUS_CLEANUP_METHOD,
                    "consensus_mode": str(cleanup.mode),
                },
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(component_rows))


def plan_consensus_bridge_splits(
    predicted_track_matrix: Any,
    *,
    diagnostics_by_edge: Mapping[tuple[int, int, int], Any],
    support_counts: Mapping[Edge, int],
    config: ConsensusSplitConfig | None = None,
    track_ids: Sequence[int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Plan conservative splits for risky, unstable bridges.

    ``diagnostics_by_edge`` uses ``(session, source_roi, target_roi)`` keys in
    the same ROI-index space as ``predicted_track_matrix``. ``support_counts``
    uses ``(session, next_session, source_roi, target_roi)`` keys as returned by
    the stability-cleanup helper. Candidate bridges are selected with the same
    global compatible-split optimizer as the multisplit cleanup, rather than a
    greedy ranked pass, so one central bridge cannot block two stronger
    compatible side splits.
    """

    cfg = config or ConsensusSplitConfig()
    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    ids = (
        tuple(range(predicted.shape[0]))
        if track_ids is None
        else tuple(int(track_id) for track_id in track_ids)
    )
    if len(ids) != predicted.shape[0]:
        raise ValueError("track_ids must have one entry per predicted track")
    split_plan: dict[int, tuple[int, ...]] = {}
    for row_index, row in enumerate(predicted):
        splits = _track_split_indices(row, diagnostics_by_edge, support_counts, cfg)
        if splits:
            split_plan[int(ids[row_index])] = splits
    return split_plan


def apply_consensus_bridge_splits(
    predicted_track_matrix: Any,
    split_plan: Mapping[int, Sequence[int]],
) -> np.ndarray:
    """Apply a consensus split plan to a prediction matrix."""

    return apply_ranked_bridge_splits(predicted_track_matrix, split_plan)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for consensus cleanup."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-consensus-cleanup",
        description=(
            "Run conservative Track2p-policy cleanup by splitting bridges that "
            "are both risky and unstable across nearby IoU thresholds."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        default="manual-gt",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", choices=("auto", "suite2p", "npy"), default="suite2p"
    )
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    parser.add_argument(
        "--base-iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--stability-iou-distance-thresholds",
        type=_float_tuple_arg,
        default=(10.0, 12.0, 14.0),
        help="Comma-separated IoU-distance thresholds used for stability voting.",
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument(
        "--apply-splits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-complete-track", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--consensus-mode",
        choices=CONSENSUS_MODES,
        default="risk-and-stability",
    )
    parser.add_argument("--threshold-margin-scale", type=float, default=0.10)
    parser.add_argument("--competition-margin-scale", type=float, default=0.20)
    parser.add_argument("--area-ratio-floor", type=float, default=0.45)
    parser.add_argument("--centroid-distance-scale", type=float, default=4.0)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument("--max-splits-per-component", type=int, default=2)
    parser.add_argument("--min-support-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-support-votes", type=int, default=None)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--component-output", type=Path, default=None)
    parser.add_argument("--component-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy consensus cleanup CLI."""

    args = build_arg_parser().parse_args(argv)
    component_config = ComponentCleanupConfig(
        threshold_margin_scale=args.threshold_margin_scale,
        competition_margin_scale=args.competition_margin_scale,
        area_ratio_floor=args.area_ratio_floor,
        centroid_distance_scale=args.centroid_distance_scale,
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    stability_config = StabilityCleanupConfig(
        iou_distance_thresholds=tuple(args.stability_iou_distance_thresholds),
        base_iou_distance_threshold=args.base_iou_distance_threshold,
        min_support_fraction=args.min_support_fraction,
        min_support_votes=args.min_support_votes,
        min_side_observations=args.min_side_observations,
    )
    cleanup_config = ConsensusCleanupConfig(
        component=component_config,
        stability=stability_config,
        max_splits_per_component=args.max_splits_per_component,
        mode=cast(ConsensusMode, args.consensus_mode),
    )
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        plane_name=args.plane_name,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
        allow_track2p_as_reference_for_smoke_test=(
            args.allow_track2p_as_reference_for_smoke_test
        ),
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    output = run_track2p_policy_consensus_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
        cleanup_config=cleanup_config,
        apply_splits=args.apply_splits,
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.component_output is not None:
        write_component_rows(
            output.component_rows,
            output_format=cast(Literal["csv", "json"], args.component_format),
            output_path=args.component_output,
        )
    return 0


def _evaluated_prediction_rows(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    config: Track2pBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    predicted = _normalize_int_track_matrix(predicted)
    reference = _normalize_int_track_matrix(reference)
    if not config.restrict_to_reference_seed_rois:
        return predicted, reference, tuple(range(predicted.shape[0]))
    reference_seed_rois = _reference_seed_roi_set(
        reference, seed_session=config.seed_session
    )
    keep_indices = tuple(
        index
        for index, row in enumerate(predicted)
        if _valid_seed_roi(row, reference_seed_rois, seed_session=config.seed_session)
    )
    predicted_eval = predicted[np.asarray(keep_indices, dtype=int)]
    reference_eval = _filter_tracks_by_seed_rois(
        reference,
        reference_seed_rois,
        seed_session=config.seed_session,
    )
    return predicted_eval, reference_eval, keep_indices


def _track_split_indices(
    row: np.ndarray,
    diagnostics_by_edge: Mapping[tuple[int, int, int], Any],
    support_counts: Mapping[Edge, int],
    cfg: ConsensusSplitConfig,
) -> tuple[int, ...]:
    if cfg.component.require_complete_track and int(np.sum(row >= 0)) != int(row.size):
        return ()
    candidate_gains: dict[int, float] = {}
    for session_index in range(max(0, row.size - 1)):
        source = int(row[session_index])
        target = int(row[session_index + 1])
        if source < 0 or target < 0:
            continue
        diagnostic = diagnostics_by_edge.get((session_index, source, target))
        risk = edge_risk_score(diagnostic, config=cfg.component)
        support = int(
            support_counts.get((session_index, session_index + 1, source, target), 0)
        )
        risky = (
            risk >= cfg.component.split_risk_threshold
            and risk > cfg.component.split_penalty
        )
        unstable = support < int(cfg.required_support_votes)
        if _passes_mode(risky, unstable, cfg.mode):
            gain = _consensus_split_gain(risk, support, cfg)
            split_index = int(session_index)
            candidate_gains[split_index] = max(
                float(gain), candidate_gains.get(split_index, -float("inf"))
            )

    return _select_optimal_split_indices(
        row,
        candidate_gains,
        max_splits=int(cfg.max_splits_per_component),
        min_observations=int(cfg.component.min_side_observations),
    )


def _consensus_split_gain(
    risk: float,
    support: int,
    cfg: ConsensusSplitConfig,
) -> float:
    """Return the optimization gain for one consensus bridge split."""

    risk_gain = float(risk) - float(cfg.component.split_penalty)
    support_deficit = max(0, int(cfg.required_support_votes) - int(support))
    if support_deficit > 0:
        support_bonus = float(support_deficit) * max(
            float(cfg.component.split_risk_threshold), 1.0
        )
        return float(support_bonus + max(risk_gain, 0.0))
    return float(risk_gain)


def _passes_mode(risky: bool, unstable: bool, mode: ConsensusMode) -> bool:
    if mode == "risk-and-stability":
        return bool(risky and unstable)
    if mode == "risk-or-stability":
        return bool(risky or unstable)
    if mode == "risk-only":
        return bool(risky)
    if mode == "stability-only":
        return bool(unstable)
    raise ValueError(f"unsupported consensus mode: {mode}")


def _enrich_component_rows_with_split_plan(
    rows: Sequence[Mapping[str, float | int | str]],
    split_plan: Mapping[int, Sequence[int]],
    *,
    support_counts: Mapping[Edge, int],
    apply_splits: bool,
) -> list[dict[str, float | int | str]]:
    enriched: list[dict[str, float | int | str]] = []
    for row in rows:
        component_id = int(row["predicted_track_id"])
        split_indices = tuple(int(index) for index in split_plan.get(component_id, ()))
        support_values = tuple(
            _support_for_row_split(row, split_index, support_counts=support_counts)
            for split_index in split_indices
        )
        updated = dict(row)
        updated["candidate_split_count"] = int(len(split_indices))
        updated["candidate_split_sessions"] = json.dumps(list(split_indices))
        updated["candidate_split_support_votes"] = json.dumps(list(support_values))
        updated["would_split_at_weakest_edge"] = int(bool(split_indices))
        updated["applied_split"] = int(len(split_indices) if apply_splits else 0)
        enriched.append(updated)
    return enriched


def _support_for_row_split(
    row: Mapping[str, float | int | str],
    split_index: int,
    *,
    support_counts: Mapping[Edge, int],
) -> int:
    if split_index == int(row.get("weakest_bridge_session_a", -1)):
        source = int(row.get("weakest_bridge_source_roi", -1))
        target = int(row.get("weakest_bridge_target_roi", -1))
        return int(support_counts.get((split_index, split_index + 1, source, target), 0))
    return -1


def _with_metadata(
    rows: Sequence[Mapping[str, float | int | str]],
    metadata: Mapping[str, Any],
) -> list[dict[str, float | int | str]]:
    formatted = {key: _format_metadata_value(value) for key, value in metadata.items()}
    return [{**dict(row), **formatted} for row in rows]


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _float_tuple_arg(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        tokens = tuple(token.strip() for token in value.split(",") if token.strip())
        if not tokens:
            raise argparse.ArgumentTypeError("expected at least one float")
        try:
            return tuple(float(token) for token in tokens)
        except ValueError as exc:  # pragma: no cover - argparse reports this path
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return tuple(float(item) for item in value)


__all__ = (
    "ConsensusCleanupConfig",
    "ConsensusSplitConfig",
    "apply_consensus_bridge_splits",
    "plan_consensus_bridge_splits",
    "run_track2p_policy_consensus_cleanup",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
