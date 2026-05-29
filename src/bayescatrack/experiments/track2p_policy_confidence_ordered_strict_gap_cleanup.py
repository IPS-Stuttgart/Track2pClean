"""Confidence-ordered strict gap rescue after component cleanup.

The strict gated gap-rescue runner already keeps only candidate skip edges that pass
hard feature gates.  This variant changes the merge policy for the remaining
accepted candidates: when multiple accepted suffixes compete for the same ROI
observation, the candidate with the largest gate slack is applied first.  That
prevents an accepted-but-barely-passing suffix from blocking a stronger accepted
suffix simply because it was discovered earlier in the candidate track matrix.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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
from bayescatrack.experiments.track2p_emulation_benchmark import (
    ThresholdMethod,
    emulate_track2p_tracks,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentAuditOutput,
    ComponentCleanupConfig,
    _normalize_int_track_matrix,
)
from bayescatrack.experiments.track2p_policy_gap_edge_audit import (
    GapAuditEdge,
    GapEdgeFeature,
    _cell_probability,
    _component_cleanup_prediction,
    _edge_feature_index,
)
from bayescatrack.experiments.track2p_policy_strict_gated_gap_cleanup import (
    STRICT_GATED_GAP_DEFAULT_MAX_GAP,
    StrictGapCandidate,
    StrictGapGateConfig,
    _candidate_rows,
    _merge_candidate_row,
    _observation_counter,
    _seed_rois,
    strict_gated_gap_candidates,
    write_strict_gap_rows,
)

TRACK2P_POLICY_CONFIDENCE_ORDERED_STRICT_GAP_CLEANUP_METHOD = (
    "track2p-policy-confidence-ordered-strict-gated-gap-cleanup"
)
CandidatePriorityKey = tuple[float, float, float, float, float, float]


def run_track2p_policy_confidence_ordered_strict_gated_gap_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int = STRICT_GATED_GAP_DEFAULT_MAX_GAP,
    cleanup_config: ComponentCleanupConfig | None = None,
    gate_config: StrictGapGateConfig | None = None,
) -> ComponentAuditOutput:
    """Run component cleanup, then merge gated gap candidates by confidence."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
        max_gap=max_gap,
    )
    if int(policy_config.max_gap) < 1:
        raise ValueError("max_gap must be at least 1")
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    gate_config = gate_config or StrictGapGateConfig()
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    edge_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy confidence-ordered strict gap cleanup requires "
                "independent manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        base_full = _component_cleanup_prediction(
            sessions,
            reference_tracks,
            config=policy_config,
            cleanup_config=cleanup_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
        )
        candidate_full = _normalize_int_track_matrix(
            emulate_track2p_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                max_gap=int(policy_config.max_gap),
            )
        )
        feature_index = _edge_feature_index(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            max_gap=int(policy_config.max_gap),
        )
        candidates = strict_gated_gap_candidates(
            base_full,
            candidate_full,
            sessions=sessions,
            feature_index=feature_index,
            gate_config=gate_config,
            max_gap=int(policy_config.max_gap),
            seed_rois=_seed_rois(
                reference_tracks, seed_session=policy_config.seed_session
            ),
            seed_session=policy_config.seed_session,
        )
        cleaned, applied_candidate_indices = (
            apply_confidence_ordered_strict_gated_gap_candidates_with_report(
                base_full,
                candidate_full,
                candidates,
                sessions=sessions,
                feature_index=feature_index,
                gate_config=gate_config,
                seed_session=policy_config.seed_session,
            )
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        gate_accepted_edges = int(sum(candidate.accepted for candidate in candidates))
        applied_edges = int(len(applied_candidate_indices))
        scores = {
            **scores,
            "track2p_policy_variant": (
                TRACK2P_POLICY_CONFIDENCE_ORDERED_STRICT_GAP_CLEANUP_METHOD
            ),
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_policy_max_gap": int(policy_config.max_gap),
            "track2p_strict_gap_candidate_edges": int(len(candidates)),
            "track2p_strict_gap_gate_accepted_edges": gate_accepted_edges,
            "track2p_strict_gap_accepted_edges": applied_edges,
            "track2p_strict_gap_applied_edges": applied_edges,
            "track2p_strict_gap_gate_rejected_edges": int(
                len(candidates) - gate_accepted_edges
            ),
            "track2p_strict_gap_merge_rejected_edges": int(
                gate_accepted_edges - applied_edges
            ),
            "track2p_strict_gap_merge_policy": "confidence-order",
            "track2p_strict_gap_length": int(gate_config.gap_length),
            "track2p_strict_gap_min_area_ratio": float(gate_config.min_area_ratio),
            "track2p_strict_gap_min_cell_probability": float(
                gate_config.min_cell_probability
            ),
            "track2p_strict_gap_max_registered_iou": float(
                gate_config.max_registered_iou
            ),
            "track2p_strict_gap_min_row_margin": float(gate_config.min_row_margin),
            "track2p_strict_gap_min_column_margin": float(
                gate_config.min_column_margin
            ),
            "track2p_strict_gap_min_threshold_margin": float(
                gate_config.min_threshold_margin
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=(
                    "Track2p-policy component cleanup + confidence-ordered strict "
                    "gap rescue"
                ),
                method=cast(
                    Any, TRACK2P_POLICY_CONFIDENCE_ORDERED_STRICT_GAP_CLEANUP_METHOD
                ),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        edge_rows.extend(
            _candidate_rows_with_priority(
                subject_dir.name,
                candidates,
                sessions=sessions,
                feature_index=feature_index,
                gate_config=gate_config,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                cell_probability_threshold=float(
                    policy_config.cell_probability_threshold
                ),
                transform_type=policy_config.transform_type,
                max_gap=int(policy_config.max_gap),
                applied_candidate_indices=applied_candidate_indices,
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(edge_rows))


def apply_confidence_ordered_strict_gated_gap_candidates(
    base_tracks: np.ndarray,
    candidate_tracks: np.ndarray,
    candidates: Sequence[StrictGapCandidate],
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
    seed_session: int = 0,
) -> np.ndarray:
    """Merge accepted strict-gap candidates in descending confidence order."""

    output, _ = apply_confidence_ordered_strict_gated_gap_candidates_with_report(
        base_tracks,
        candidate_tracks,
        candidates,
        sessions=sessions,
        feature_index=feature_index,
        gate_config=gate_config,
        seed_session=seed_session,
    )
    return output


def apply_confidence_ordered_strict_gated_gap_candidates_with_report(
    base_tracks: np.ndarray,
    candidate_tracks: np.ndarray,
    candidates: Sequence[StrictGapCandidate],
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
    seed_session: int = 0,
) -> tuple[np.ndarray, frozenset[int]]:
    """Merge accepted candidates by priority and report original indices applied."""

    output = _normalize_int_track_matrix(base_tracks).copy()
    candidate_matrix = _normalize_int_track_matrix(candidate_tracks)
    observation_counts = _observation_counter(output)
    applied: set[int] = set()
    for candidate_index in confidence_ordered_candidate_indices(
        candidates,
        sessions=sessions,
        feature_index=feature_index,
        gate_config=gate_config,
    ):
        candidate = candidates[candidate_index]
        if not candidate.accepted:
            continue
        if not 0 <= candidate.candidate_track_id < candidate_matrix.shape[0]:
            continue
        candidate_row = candidate_matrix[candidate.candidate_track_id]
        updated = _merge_candidate_row(
            output,
            candidate_row,
            candidate.edge,
            observation_counts=observation_counts,
            seed_session=seed_session,
        )
        if updated is not output:
            applied.add(int(candidate_index))
        output = updated
    return output, frozenset(applied)


def confidence_ordered_candidate_indices(
    candidates: Sequence[StrictGapCandidate],
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
) -> tuple[int, ...]:
    """Return candidate indices sorted by descending gate slack/evidence strength."""

    return tuple(
        sorted(
            range(len(candidates)),
            key=lambda index: strict_gap_candidate_priority_key(
                candidates[index],
                sessions=sessions,
                feature_index=feature_index,
                gate_config=gate_config,
            ),
            reverse=True,
        )
    )


def strict_gap_candidate_priority_key(
    candidate: StrictGapCandidate,
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
) -> CandidatePriorityKey:
    """Return a sortable confidence key for an already-gated gap candidate.

    The primary term is the smallest slack across the hard gate inequalities.  A
    candidate that barely passes any one gate is therefore applied after a
    candidate that passes all gates comfortably.  The remaining terms are
    deterministic tie-breakers based on the same local evidence.
    """

    if not candidate.accepted:
        return _rejected_priority_key(candidate)
    feature = feature_index.get(candidate.edge)
    if feature is None:
        return _rejected_priority_key(candidate)
    margins = _candidate_gate_margins(
        candidate,
        feature,
        sessions=sessions,
        gate_config=gate_config,
    )
    bottleneck = min(margins)
    return (
        bottleneck,
        margins[5],  # threshold margin slack
        margins[0],  # area-ratio slack
        min(margins[3], margins[4]),  # assignment-competition slack
        -_finite_or_positive_infinity(feature.centroid_distance),
        -float(candidate.candidate_track_id),
    )


def _candidate_gate_margins(
    candidate: StrictGapCandidate,
    feature: GapEdgeFeature,
    *,
    sessions: Sequence[Track2pSession],
    gate_config: StrictGapGateConfig,
) -> tuple[float, float, float, float, float, float]:
    session_a, session_b, roi_a, roi_b = candidate.edge
    probability_a = _cell_probability(sessions[session_a], int(roi_a))
    probability_b = _cell_probability(sessions[session_b], int(roi_b))
    return (
        _finite_or_negative_infinity(feature.area_ratio - gate_config.min_area_ratio),
        _finite_or_negative_infinity(
            min(probability_a, probability_b) - gate_config.min_cell_probability
        ),
        _finite_or_negative_infinity(
            gate_config.max_registered_iou - feature.registered_iou
        ),
        _finite_or_negative_infinity(feature.row_margin - gate_config.min_row_margin),
        _finite_or_negative_infinity(
            feature.column_margin - gate_config.min_column_margin
        ),
        _finite_or_negative_infinity(
            feature.threshold_margin - gate_config.min_threshold_margin
        ),
    )


def _candidate_rows_with_priority(
    subject: str,
    candidates: Sequence[StrictGapCandidate],
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cell_probability_threshold: float,
    transform_type: str,
    max_gap: int,
    applied_candidate_indices: frozenset[int] | set[int] = frozenset(),
) -> list[dict[str, float | int | str]]:
    rows = _candidate_rows(
        subject,
        candidates,
        sessions=sessions,
        feature_index=feature_index,
        gate_config=gate_config,
        threshold_method=threshold_method,
        iou_distance_threshold=iou_distance_threshold,
        cell_probability_threshold=cell_probability_threshold,
        transform_type=transform_type,
        max_gap=max_gap,
        applied_candidate_indices=applied_candidate_indices,
    )
    priority_order = confidence_ordered_candidate_indices(
        candidates,
        sessions=sessions,
        feature_index=feature_index,
        gate_config=gate_config,
    )
    priority_rank_by_index = {
        candidate_index: rank
        for rank, candidate_index in enumerate(priority_order, start=1)
    }
    output: list[dict[str, float | int | str]] = []
    for candidate_index, row in enumerate(rows):
        priority_key = strict_gap_candidate_priority_key(
            candidates[candidate_index],
            sessions=sessions,
            feature_index=feature_index,
            gate_config=gate_config,
        )
        output.append(
            {
                **dict(row),
                "merge_priority_rank": int(priority_rank_by_index[candidate_index]),
                "merge_priority_bottleneck_margin": _finite_or_nan(priority_key[0]),
                "merge_priority_threshold_margin": _finite_or_nan(priority_key[1]),
                "merge_priority_area_ratio_margin": _finite_or_nan(priority_key[2]),
                "merge_priority_assignment_margin": _finite_or_nan(priority_key[3]),
            }
        )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the confidence-ordered strict gap cleanup CLI parser."""

    parser = argparse.ArgumentParser(
        prog=(
            "bayescatrack benchmark "
            "track2p-policy-confidence-ordered-strict-gated-gap-cleanup"
        ),
        description=(
            "Run Track2p-policy component cleanup, then admit only hard-gated "
            "gap-rescue candidate edges and merge accepted candidates by "
            "descending gate slack."
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
    parser.add_argument("--max-gap", type=int, default=STRICT_GATED_GAP_DEFAULT_MAX_GAP)
    parser.add_argument(
        "--gate-gap-length",
        type=int,
        default=StrictGapGateConfig().gap_length,
    )
    parser.add_argument(
        "--gate-min-area-ratio",
        type=float,
        default=StrictGapGateConfig().min_area_ratio,
    )
    parser.add_argument(
        "--gate-min-cell-probability",
        type=float,
        default=StrictGapGateConfig().min_cell_probability,
    )
    parser.add_argument(
        "--gate-max-registered-iou",
        type=float,
        default=StrictGapGateConfig().max_registered_iou,
    )
    parser.add_argument(
        "--gate-min-row-margin",
        type=float,
        default=StrictGapGateConfig().min_row_margin,
    )
    parser.add_argument(
        "--gate-min-column-margin",
        type=float,
        default=StrictGapGateConfig().min_column_margin,
    )
    parser.add_argument(
        "--gate-min-threshold-margin",
        type=float,
        default=StrictGapGateConfig().min_threshold_margin,
    )
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument("--edge-output", type=Path, default=None)
    parser.add_argument("--edge-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the confidence-ordered strict gap cleanup CLI."""

    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    gate_config = StrictGapGateConfig(
        gap_length=args.gate_gap_length,
        min_area_ratio=args.gate_min_area_ratio,
        min_cell_probability=args.gate_min_cell_probability,
        max_registered_iou=args.gate_max_registered_iou,
        min_row_margin=args.gate_min_row_margin,
        min_column_margin=args.gate_min_column_margin,
        min_threshold_margin=args.gate_min_threshold_margin,
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
    output = run_track2p_policy_confidence_ordered_strict_gated_gap_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        max_gap=int(args.max_gap),
        cleanup_config=cleanup_config,
        gate_config=gate_config,
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.edge_output is not None:
        write_strict_gap_rows(
            output.component_rows,
            args.edge_output,
            output_format=cast(Literal["csv", "json"], args.edge_format),
        )
    return 0


def _rejected_priority_key(candidate: StrictGapCandidate) -> CandidatePriorityKey:
    return (
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
        -float(candidate.candidate_track_id),
    )


def _finite_or_negative_infinity(value: float) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("-inf")


def _finite_or_positive_infinity(value: float) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("inf")


def _finite_or_nan(value: float) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("nan")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
