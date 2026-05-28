"""Gap-aware weakest-bridge cleanup for Track2p-policy gap rescue.

The existing component cleanup scores only adjacent occupied session pairs.  That
misses the failure mode created by gap rescue: a row such as ``[seed, -1,
suffix]`` contains a direct rescued bridge from session 0 to session 2, but no
adjacent occupied bridge at session 0->1 or 1->2.  This module evaluates bridges
between consecutive *observations* instead, so weak rescued suffixes can be split
without disabling useful gap rescue globally.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
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
from bayescatrack.experiments.track2p_emulation_benchmark import emulate_track2p_tracks
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
    _mark_applied_splits,
    _normalize_int_track_matrix,
    _split_gain,
    _split_observation_counts,
    apply_weakest_bridge_splits,
    edge_risk_score,
    write_component_rows,
)
from bayescatrack.experiments.track2p_policy_gap_component_cleanup import (
    TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP,
    _default_gap_cleanup_config,
    _format_metadata_value,
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
    Track2pPolicyPruneConfig,
    _thresholded_pruned_hungarian_links,
)

TRACK2P_POLICY_GAP_BRIDGE_CLEANUP_METHOD = "track2p-policy-gap-bridge-cleanup"
GapEdgeKey = tuple[int, int, int, int]


@dataclass(frozen=True)
class GapComponentEdge:
    """One bridge between consecutive observations in a predicted component."""

    session_a: int
    session_b: int
    source_roi: int
    target_roi: int
    diagnostic: Track2pPolicyLinkDiagnostic | None
    risk: float


def run_track2p_policy_gap_bridge_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    apply_splits: bool = True,
) -> ComponentAuditOutput:
    """Run gap rescue and split weak bridges between consecutive observations."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
        max_gap=max_gap,
    )
    if int(policy_config.max_gap) < 1:
        raise ValueError("max_gap must be at least 1")
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    cleanup_config = cleanup_config or _default_gap_cleanup_config()
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
                "Track2p-policy gap bridge cleanup requires independent manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        predicted_full = _normalize_int_track_matrix(
            emulate_track2p_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                max_gap=int(policy_config.max_gap),
            )
        )
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        predicted_eval, _reference_eval, evaluated_track_ids = (
            _evaluated_prediction_rows(
                predicted_full,
                reference_tracks,
                config=policy_config,
            )
        )
        diagnostic_by_edge = gap_link_diagnostics_by_suite2p_edge(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            max_gap=int(policy_config.max_gap),
            prune_config=_no_prune_config(),
        )
        audit_rows = gap_bridge_component_audit_rows(
            predicted_eval,
            diagnostic_by_edge,
            subject=subject_dir.name,
            config=cleanup_config,
            track_ids=evaluated_track_ids,
        )
        subject_rows = _mark_applied_splits(audit_rows, apply_splits=apply_splits)
        cleaned = (
            apply_weakest_bridge_splits(predicted_full, subject_rows)
            if apply_splits
            else predicted_full
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        scores = _with_scores_metadata(
            scores,
            threshold_method=threshold_method,
            iou_distance_threshold=iou_distance_threshold,
            cleanup_config=cleanup_config,
            max_gap=policy_config.max_gap,
            cell_probability_threshold=policy_config.cell_probability_threshold,
            transform_type=policy_config.transform_type,
            apply_splits=apply_splits,
            component_rows=subject_rows,
        )
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=(
                    "Track2p-policy gap-rescue gap-bridge split"
                    if apply_splits
                    else "Track2p-policy gap-rescue gap-bridge audit"
                ),
                method=cast(Any, TRACK2P_POLICY_GAP_BRIDGE_CLEANUP_METHOD),
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
                    "max_gap": int(policy_config.max_gap),
                    "cleanup_method": TRACK2P_POLICY_GAP_BRIDGE_CLEANUP_METHOD,
                },
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(component_rows))


def gap_link_diagnostics_by_suite2p_edge(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    max_gap: int,
    prune_config: Track2pPolicyPruneConfig | None = None,
) -> dict[GapEdgeKey, Track2pPolicyLinkDiagnostic]:
    """Return diagnostics keyed by ``(session_a, session_b, roi_a, roi_b)``."""

    sessions = tuple(sessions)
    max_gap = int(max_gap)
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    prune = prune_config or _no_prune_config()
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    diagnostics_by_edge: dict[GapEdgeKey, Track2pPolicyLinkDiagnostic] = {}
    for session_a in range(max(0, len(sessions) - 1)):
        max_step = min(max_gap, len(sessions) - 1 - session_a)
        for step in range(1, max_step + 1):
            session_b = session_a + step
            _, diagnostics = _thresholded_pruned_hungarian_links(
                sessions[session_a],
                sessions[session_b],
                session_index=session_a,
                transform_type=transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold) * float(step),
                prune_config=prune,
            )
            source_indices = roi_indices_by_session[session_a]
            target_indices = roi_indices_by_session[session_b]
            for diagnostic in diagnostics:
                key = (
                    int(session_a),
                    int(session_b),
                    int(source_indices[diagnostic.local_roi_a]),
                    int(target_indices[diagnostic.local_roi_b]),
                )
                diagnostics_by_edge[key] = diagnostic
    return diagnostics_by_edge


def gap_bridge_component_audit_rows(
    predicted_track_matrix: Any,
    diagnostic_by_edge: Mapping[GapEdgeKey, Track2pPolicyLinkDiagnostic],
    *,
    subject: str = "",
    config: ComponentCleanupConfig | None = None,
    track_ids: Sequence[int] | None = None,
) -> list[dict[str, float | int | str]]:
    """Audit bridges between consecutive observations, including rescued gaps."""

    cfg = config or _default_gap_cleanup_config()
    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    ids = (
        tuple(range(predicted.shape[0]))
        if track_ids is None
        else tuple(int(track_id) for track_id in track_ids)
    )
    if len(ids) != predicted.shape[0]:
        raise ValueError("track_ids must have one entry per predicted track")

    rows: list[dict[str, float | int | str]] = []
    for row_index, track in enumerate(predicted):
        component_id = ids[row_index]
        edges = _gap_component_edges(track, diagnostic_by_edge, config=cfg)
        weakest = max(edges, key=lambda edge: edge.risk, default=None)
        valid_observations = int(np.sum(track >= 0))
        is_complete_track = bool(valid_observations == int(predicted.shape[1]))
        split_index = -1 if weakest is None else int(weakest.session_a)
        left_observations, right_observations = _split_observation_counts(
            track, split_index
        )
        split_gain = (
            _split_gain(
                weakest.risk if weakest is not None else 0.0,
                left_observations=left_observations,
                right_observations=right_observations,
                config=cfg,
            )
            if weakest is not None
            else -float(cfg.split_penalty)
        )
        would_split = bool(
            weakest is not None
            and weakest.risk >= cfg.split_risk_threshold
            and left_observations >= cfg.min_side_observations
            and right_observations >= cfg.min_side_observations
            and (is_complete_track or not cfg.require_complete_track)
            and split_gain > 0.0
        )
        rows.append(
            {
                "subject": subject,
                "predicted_track_id": int(component_id),
                "n_sessions": valid_observations,
                "total_sessions": int(predicted.shape[1]),
                "n_rois": valid_observations,
                "n_edges": int(len(edges)),
                "is_complete_track": int(is_complete_track),
                "component_score": float(weakest.risk if weakest else 0.0),
                "component_risk_sum": float(sum(edge.risk for edge in edges)),
                "weakest_bridge_session_a": split_index,
                "weakest_bridge_session_b": (
                    int(weakest.session_b) if weakest is not None else -1
                ),
                "weakest_bridge_source_roi": (
                    int(weakest.source_roi) if weakest is not None else -1
                ),
                "weakest_bridge_target_roi": (
                    int(weakest.target_roi) if weakest is not None else -1
                ),
                "weakest_bridge_gap": (
                    int(weakest.session_b - weakest.session_a)
                    if weakest is not None
                    else 0
                ),
                "weakest_bridge_risk": float(weakest.risk if weakest else 0.0),
                "split_gain": float(split_gain),
                "would_split_at_weakest_edge": int(would_split),
                "applied_split": 0,
                "complete_track_status_against_gt": (
                    "incomplete" if not is_complete_track else "not_evaluated"
                ),
                "pairwise_tp_edges": 0,
                "pairwise_fp_edges": 0,
                "pairwise_fn_edges": 0,
            }
        )
    return rows


def _gap_component_edges(
    track: np.ndarray,
    diagnostic_by_edge: Mapping[GapEdgeKey, Track2pPolicyLinkDiagnostic],
    *,
    config: ComponentCleanupConfig,
) -> list[GapComponentEdge]:
    observed_sessions = tuple(int(index) for index in np.flatnonzero(track >= 0))
    edges: list[GapComponentEdge] = []
    for session_a, session_b in zip(
        observed_sessions[:-1], observed_sessions[1:], strict=True
    ):
        source_roi = int(track[session_a])
        target_roi = int(track[session_b])
        diagnostic = diagnostic_by_edge.get(
            (int(session_a), int(session_b), source_roi, target_roi)
        )
        edges.append(
            GapComponentEdge(
                session_a=int(session_a),
                session_b=int(session_b),
                source_roi=source_roi,
                target_roi=target_roi,
                diagnostic=diagnostic,
                risk=edge_risk_score(diagnostic, config=config),
            )
        )
    return edges


def _with_scores_metadata(
    scores: Mapping[str, float | int | str],
    *,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cleanup_config: ComponentCleanupConfig,
    max_gap: int,
    cell_probability_threshold: float,
    transform_type: str,
    apply_splits: bool,
    component_rows: Sequence[Mapping[str, float | int | str]],
) -> dict[str, float | int | str]:
    candidate_splits = int(
        sum(int(row["would_split_at_weakest_edge"]) for row in component_rows)
    )
    applied_splits = int(
        sum(int(row["applied_split"]) for row in component_rows) if apply_splits else 0
    )
    return {
        **dict(scores),
        "track2p_policy_threshold_method": str(threshold_method),
        "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
        "track2p_policy_cell_probability_threshold": float(cell_probability_threshold),
        "track2p_policy_transform_type": str(transform_type),
        "track2p_policy_max_gap": int(max_gap),
        "track2p_component_apply_splits": int(apply_splits),
        "track2p_component_candidate_splits": candidate_splits,
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
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for gap-aware bridge cleanup."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-gap-bridge-cleanup",
        description=(
            "Run Track2p-policy gap rescue followed by weakest observed-bridge cleanup."
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
        "--max-gap",
        type=int,
        default=TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP,
    )
    parser.add_argument(
        "--apply-splits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--threshold-margin-scale", type=float, default=0.10)
    parser.add_argument("--competition-margin-scale", type=float, default=0.20)
    parser.add_argument("--area-ratio-floor", type=float, default=0.45)
    parser.add_argument("--centroid-distance-scale", type=float, default=4.0)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=1)
    parser.add_argument(
        "--require-complete-track", action=argparse.BooleanOptionalAction, default=False
    )
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
    """Run the Track2p-policy gap-aware bridge cleanup CLI."""

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
        max_gap=args.max_gap,
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
    output = run_track2p_policy_gap_bridge_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
        max_gap=args.max_gap,
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
            args.component_output,
            output_format=cast(Literal["csv", "json"], args.component_format),
        )
    return 0


def _roi_indices(session: Track2pSession) -> np.ndarray:
    roi_indices = session.plane_data.roi_indices
    if roi_indices is None:
        return np.arange(session.plane_data.n_rois, dtype=int)
    return np.asarray(roi_indices, dtype=int)


def _with_metadata(
    rows: Sequence[Mapping[str, float | int | str]],
    metadata: Mapping[str, Any],
) -> list[dict[str, float | int | str]]:
    formatted = {key: _format_metadata_value(value) for key, value in metadata.items()}
    return [{**dict(row), **formatted} for row in rows]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
