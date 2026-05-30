"""Conflict-augmented component cleanup for Track2p-policy rows.

The strongest Track2p-policy baseline already uses weakest-bridge component
splitting.  Its audit rows also expose structural evidence such as duplicate
predicted edges and reused ROI observations, but the default split decision only
uses local bridge risk.  This runner turns those structural conflict flags into a
small reference-free risk bonus, so complete predicted components that are both
locally weak and structurally suspicious can be split before scoring.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentAuditOutput,
    ComponentCleanupConfig,
    _evaluated_prediction_rows,
    _no_prune_config,
    _normalize_int_track_matrix,
    apply_weakest_bridge_splits,
    component_audit_rows,
    write_component_rows,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_CONFLICT_COMPONENT_CLEANUP_METHOD = (
    "track2p-policy-conflict-component-cleanup"
)


@dataclass(frozen=True)
class ConflictAugmentedCleanupConfig:
    """Reference-free structural bonuses for component split decisions."""

    component_config: ComponentCleanupConfig = field(
        default_factory=ComponentCleanupConfig
    )
    conflicting_observation_bonus: float = 0.50
    duplicate_edge_bonus: float = 0.50
    min_base_risk: float = 0.25

    def __post_init__(self) -> None:
        _require_nonnegative(
            self.conflicting_observation_bonus,
            name="conflicting_observation_bonus",
        )
        _require_nonnegative(self.duplicate_edge_bonus, name="duplicate_edge_bonus")
        _require_nonnegative(self.min_base_risk, name="min_base_risk")


def run_track2p_policy_conflict_component_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    conflict_config: ConflictAugmentedCleanupConfig | None = None,
    apply_splits: bool = True,
) -> ComponentAuditOutput:
    """Run component cleanup with duplicate/conflict-augmented split scoring."""

    if conflict_config is None:
        conflict_config = ConflictAugmentedCleanupConfig(
            component_config=cleanup_config or ComponentCleanupConfig()
        )
    elif cleanup_config is not None:
        conflict_config = replace(conflict_config, component_config=cleanup_config)
    cleanup_config = conflict_config.component_config

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
                "Track2p-policy conflict component cleanup requires independent "
                "manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=_no_prune_config(),
        )
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        predicted_full = _normalize_int_track_matrix(prediction.tracks)
        predicted_eval, reference_eval, evaluated_track_ids = (
            _evaluated_prediction_rows(
                predicted_full,
                reference_tracks,
                config=policy_config,
            )
        )
        audit_rows = component_audit_rows(
            predicted_eval,
            reference_eval,
            sessions=sessions,
            diagnostics=prediction.diagnostics,
            subject=subject_dir.name,
            config=cleanup_config,
            track_ids=evaluated_track_ids,
            seed_session=policy_config.seed_session,
        )
        subject_rows = mark_conflict_augmented_splits(
            audit_rows,
            config=conflict_config,
            apply_splits=apply_splits,
        )
        cleaned = (
            apply_weakest_bridge_splits(predicted_full, subject_rows)
            if apply_splits
            else predicted_full
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        applied_splits = int(
            sum(int(row["applied_split"]) for row in subject_rows)
            if apply_splits
            else 0
        )
        candidate_splits = int(
            sum(int(row["would_split_at_weakest_edge"]) for row in subject_rows)
        )
        extra_splits = int(
            sum(int(row["conflict_augmented_extra_split"]) for row in subject_rows)
        )
        scores = {
            **scores,
            "track2p_policy_variant": TRACK2P_POLICY_CONFLICT_COMPONENT_CLEANUP_METHOD,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_component_apply_splits": int(apply_splits),
            "track2p_component_candidate_splits": candidate_splits,
            "track2p_component_conflict_extra_splits": extra_splits,
            "track2p_component_applied_splits": applied_splits,
            "track2p_component_split_risk_threshold": float(
                cleanup_config.split_risk_threshold
            ),
            "track2p_component_split_penalty": float(cleanup_config.split_penalty),
            "track2p_component_min_side_observations": int(
                cleanup_config.min_side_observations
            ),
            "track2p_component_require_complete_track": int(
                cleanup_config.require_complete_track
            ),
            "track2p_component_conflicting_observation_bonus": float(
                conflict_config.conflicting_observation_bonus
            ),
            "track2p_component_duplicate_edge_bonus": float(
                conflict_config.duplicate_edge_bonus
            ),
            "track2p_component_min_base_risk": float(conflict_config.min_base_risk),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant="Track2p-policy conflict-augmented component split",
                method=cast(Any, TRACK2P_POLICY_CONFLICT_COMPONENT_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        component_rows.extend(
            _with_metadata(
                subject_rows,
                {
                    "threshold_method": str(threshold_method),
                    "iou_distance_threshold": float(iou_distance_threshold),
                    "cell_probability_threshold": float(
                        policy_config.cell_probability_threshold
                    ),
                    "transform_type": str(policy_config.transform_type),
                    "conflicting_observation_bonus": float(
                        conflict_config.conflicting_observation_bonus
                    ),
                    "duplicate_edge_bonus": float(conflict_config.duplicate_edge_bonus),
                    "min_base_risk": float(conflict_config.min_base_risk),
                    "cleanup_method": TRACK2P_POLICY_CONFLICT_COMPONENT_CLEANUP_METHOD,
                },
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(component_rows))


def mark_conflict_augmented_splits(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    config: ConflictAugmentedCleanupConfig | None = None,
    apply_splits: bool = True,
) -> list[dict[str, float | int | str]]:
    """Mark weakest-bridge splits after adding structural conflict bonuses."""

    config = config or ConflictAugmentedCleanupConfig()
    marked: list[dict[str, float | int | str]] = []
    for row in rows:
        updated = dict(row)
        base_score = _as_float(row.get("component_score"), default=0.0)
        conflict_bonus = (
            float(config.conflicting_observation_bonus)
            if _as_int(row.get("n_conflicting_edges"), default=0) > 0
            else 0.0
        )
        duplicate_bonus = (
            float(config.duplicate_edge_bonus)
            if _as_int(row.get("n_same_predicted_edges"), default=0) > 0
            else 0.0
        )
        augmented_bonus = conflict_bonus + duplicate_bonus
        augmented_score = base_score + augmented_bonus
        original_candidate = _as_bool_int(row.get("would_split_at_weakest_edge"))
        extra_candidate = _conflict_augmented_extra_candidate(
            row,
            base_score=base_score,
            augmented_score=augmented_score,
            augmented_bonus=augmented_bonus,
            config=config,
        )
        candidate = bool(original_candidate or extra_candidate)
        updated["conflict_augmented_bonus"] = float(augmented_bonus)
        updated["conflict_augmented_component_score"] = float(augmented_score)
        updated["conflict_augmented_extra_split"] = int(extra_candidate)
        updated["would_split_with_conflict_augmentation"] = int(candidate)
        updated["applied_split"] = int(candidate) if apply_splits else 0
        marked.append(updated)
    return marked


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for conflict-augmented component cleanup."""

    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "bayescatrack.experiments.track2p_policy_conflict_component_cleanup"
        ),
        description=(
            "Run Track2p-policy component cleanup with duplicate/conflict "
            "structural split bonuses."
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
        "--iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
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
        "--apply-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply marked weakest-bridge splits before scoring.",
    )
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only split complete predicted tracks.",
    )
    parser.add_argument("--threshold-margin-scale", type=float, default=0.10)
    parser.add_argument("--competition-margin-scale", type=float, default=0.20)
    parser.add_argument("--area-ratio-floor", type=float, default=0.45)
    parser.add_argument("--centroid-distance-scale", type=float, default=4.0)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument("--conflicting-observation-bonus", type=float, default=0.50)
    parser.add_argument("--duplicate-edge-bonus", type=float, default=0.50)
    parser.add_argument("--min-base-risk", type=float, default=0.25)
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
    """Run the conflict-augmented Track2p-policy component cleanup CLI."""

    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        threshold_margin_scale=args.threshold_margin_scale,
        competition_margin_scale=args.competition_margin_scale,
        area_ratio_floor=args.area_ratio_floor,
        centroid_distance_scale=args.centroid_distance_scale,
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    conflict_config = ConflictAugmentedCleanupConfig(
        component_config=cleanup_config,
        conflicting_observation_bonus=args.conflicting_observation_bonus,
        duplicate_edge_bonus=args.duplicate_edge_bonus,
        min_base_risk=args.min_base_risk,
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
    output = run_track2p_policy_conflict_component_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        conflict_config=conflict_config,
        apply_splits=bool(args.apply_splits),
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
            args.component_output,
            output_format=cast(Literal["csv", "json"], args.component_format),
        )
    return 0


def _conflict_augmented_extra_candidate(
    row: Mapping[str, float | int | str],
    *,
    base_score: float,
    augmented_score: float,
    augmented_bonus: float,
    config: ConflictAugmentedCleanupConfig,
) -> bool:
    if _as_bool_int(row.get("would_split_at_weakest_edge")):
        return False
    if augmented_bonus <= 0.0:
        return False
    if base_score < float(config.min_base_risk):
        return False

    component_config = config.component_config
    if (
        not _as_bool_int(row.get("is_complete_track"))
        and bool(component_config.require_complete_track)
    ):
        return False

    split_index = _as_int(row.get("weakest_bridge_session_a"), default=-1)
    total_sessions = _as_int(
        row.get("total_sessions"),
        default=_as_int(row.get("n_sessions"), default=0),
    )
    if split_index < 0 or split_index >= total_sessions - 1:
        return False
    left_observations = split_index + 1
    right_observations = total_sessions - split_index - 1
    if left_observations < int(
        component_config.min_side_observations
    ) or right_observations < int(component_config.min_side_observations):
        return False
    if augmented_score < float(component_config.split_risk_threshold):
        return False
    return bool(augmented_score - float(component_config.split_penalty) > 0.0)


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


def _as_bool_int(value: Any) -> bool:
    return _as_int(value, default=0) != 0


def _as_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(numeric) if np.isfinite(numeric) else float(default)


def _as_int(value: Any, *, default: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(default)
    return int(numeric) if np.isfinite(numeric) else int(default)


def _require_nonnegative(value: float, *, name: str) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")


__all__ = (
    "ConflictAugmentedCleanupConfig",
    "TRACK2P_POLICY_CONFLICT_COMPONENT_CLEANUP_METHOD",
    "mark_conflict_augmented_splits",
    "run_track2p_policy_conflict_component_cleanup",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
