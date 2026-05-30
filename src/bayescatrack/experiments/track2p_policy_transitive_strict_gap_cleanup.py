"""Transitive strict gap rescue for Track2p-policy cleanup.

The one-pass strict gap cleanup can accept a high-confidence skip edge and insert
only that target observation.  If another high-confidence skip edge starts from
that newly inserted target, the one-pass candidate scan never sees it because the
source was not present in the baseline component-cleanup matrix.  This module
keeps the same hard gate and conflict checks, but iterates the candidate scan
until no newly inserted observation exposes another accepted strict gap edge.
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
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
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
    _component_cleanup_prediction,
    _observed_neighbor_edges,
)
from bayescatrack.experiments.track2p_policy_strict_gated_gap_cleanup import (
    STRICT_GATED_GAP_DEFAULT_MAX_GAP,
    StrictGapCandidate,
    StrictGapGateConfig,
    _apply_strict_gated_gap_edges_with_report,
    _candidate_rows,
    _seed_rois,
    _source_track_id,
    _valid_seed,
    strict_gap_feature_index_for_gap_length,
    strict_gap_gate_decision,
    write_strict_gap_rows,
)

TRACK2P_POLICY_TRANSITIVE_STRICT_GAP_CLEANUP_METHOD = (
    "track2p-policy-transitive-strict-gated-gap-cleanup"
)


def run_track2p_policy_transitive_strict_gated_gap_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int = STRICT_GATED_GAP_DEFAULT_MAX_GAP,
    cleanup_config: ComponentCleanupConfig | None = None,
    gate_config: StrictGapGateConfig | None = None,
    max_rounds: int | None = None,
) -> ComponentAuditOutput:
    """Run component cleanup, then iteratively insert strict gap candidates."""

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
                "Track2p-policy transitive strict gap cleanup requires independent "
                "manual GT references"
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
        feature_index = strict_gap_feature_index_for_gap_length(
            sessions,
            gap_length=int(gate_config.gap_length),
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
        )
        seed_rois = _seed_rois(reference_tracks, seed_session=policy_config.seed_session)
        cleaned, candidates, applied_candidate_indices = (
            apply_transitive_strict_gated_gap_edges_with_report(
                base_full,
                sessions=sessions,
                feature_index=feature_index,
                gate_config=gate_config,
                seed_rois=seed_rois,
                seed_session=policy_config.seed_session,
                max_rounds=max_rounds,
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
                TRACK2P_POLICY_TRANSITIVE_STRICT_GAP_CLEANUP_METHOD
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
            "track2p_strict_gap_merge_policy": "transitive-confidence-order",
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
                    "Track2p-policy component cleanup + transitive strict gap rescue"
                ),
                method=cast(Any, TRACK2P_POLICY_TRANSITIVE_STRICT_GAP_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        edge_rows.extend(
            _candidate_rows(
                subject_dir.name,
                candidates,
                sessions=sessions,
                feature_index=feature_index,
                gate_config=gate_config,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                cell_probability_threshold=float(policy_config.cell_probability_threshold),
                transform_type=policy_config.transform_type,
                max_gap=int(policy_config.max_gap),
                applied_candidate_indices=applied_candidate_indices,
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(edge_rows))


def apply_transitive_strict_gated_gap_edges(
    base_tracks: np.ndarray,
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
    seed_rois: set[int],
    seed_session: int = 0,
    max_rounds: int | None = None,
) -> np.ndarray:
    """Iteratively insert accepted strict gap edges.

    Each round considers accepted gap edges whose source ROI is visible in the
    current cleaned matrix. The same hard gate, duplicate-edge suppression, and
    ROI-observation conflict checks as the one-pass strict cleanup are retained.
    """

    output, _candidates, _applied = apply_transitive_strict_gated_gap_edges_with_report(
        base_tracks,
        sessions=sessions,
        feature_index=feature_index,
        gate_config=gate_config,
        seed_rois=seed_rois,
        seed_session=seed_session,
        max_rounds=max_rounds,
    )
    return output


def apply_transitive_strict_gated_gap_edges_with_report(
    base_tracks: np.ndarray,
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
    seed_rois: set[int],
    seed_session: int = 0,
    max_rounds: int | None = None,
) -> tuple[np.ndarray, tuple[StrictGapCandidate, ...], frozenset[int]]:
    """Return cleaned tracks, evaluated candidates, and applied candidate indices."""

    output = _normalize_int_track_matrix(base_tracks).copy()
    rounds_left = _round_limit(output, feature_index, max_rounds=max_rounds)
    seen_edges = _initial_observed_edges(
        output,
        max_gap=int(gate_config.gap_length),
        seed_rois=seed_rois,
        seed_session=seed_session,
    )
    candidates: list[StrictGapCandidate] = []
    applied: set[int] = set()

    while rounds_left > 0:
        rounds_left -= 1
        round_candidates = _next_round_candidates(
            output,
            sessions=sessions,
            feature_index=feature_index,
            gate_config=gate_config,
            seed_rois=seed_rois,
            seed_session=seed_session,
            seen_edges=seen_edges,
        )
        if not round_candidates:
            break

        round_start = len(candidates)
        candidates.extend(round_candidates)
        accepted_pairs = tuple(
            (candidate_index, candidate)
            for candidate_index, candidate in enumerate(round_candidates)
            if bool(candidate.accepted)
        )
        accepted_pairs = tuple(
            sorted(
                accepted_pairs,
                key=lambda item: _candidate_confidence(item[1], feature_index),
                reverse=True,
            )
        )
        accepted = tuple(candidate for _candidate_index, candidate in accepted_pairs)
        updated, round_applied = _apply_strict_gated_gap_edges_with_report(
            output,
            accepted,
            seed_session=seed_session,
        )
        if not round_applied:
            break

        applied.update(round_start + accepted_pairs[index][0] for index in round_applied)
        output = updated

    return output, tuple(candidates), frozenset(applied)


def _next_round_candidates(
    current_tracks: np.ndarray,
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
    seed_rois: set[int],
    seed_session: int,
    seen_edges: set[GapAuditEdge],
) -> tuple[StrictGapCandidate, ...]:
    decisions: list[StrictGapCandidate] = []
    for edge in sorted(feature_index):
        if edge in seen_edges:
            continue
        track_id = _source_track_id(
            current_tracks,
            edge,
            seed_rois=seed_rois,
            seed_session=seed_session,
        )
        if track_id is None:
            continue
        accepted, reason = strict_gap_gate_decision(
            edge,
            feature_index.get(edge),
            sessions=sessions,
            gate_config=gate_config,
        )
        decisions.append(
            StrictGapCandidate(
                edge=edge,
                candidate_track_id=int(track_id),
                accepted=accepted,
                reason=reason,
            )
        )
        seen_edges.add(edge)
    return tuple(decisions)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the transitive strict gap cleanup CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-transitive-strict-gated-gap-cleanup",
        description=(
            "Run Track2p-policy component cleanup, then iteratively admit only "
            "hard-gated gap-rescue candidate edges."
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
    parser.add_argument("--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE)
    parser.add_argument("--max-gap", type=int, default=STRICT_GATED_GAP_DEFAULT_MAX_GAP)
    parser.add_argument(
        "--gate-gap-length", type=int, default=StrictGapGateConfig().gap_length
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
    parser.add_argument("--max-rounds", type=int, default=None)
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
    """Run the transitive strict gated gap cleanup CLI."""

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
    output = run_track2p_policy_transitive_strict_gated_gap_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        max_gap=int(args.max_gap),
        cleanup_config=cleanup_config,
        gate_config=gate_config,
        max_rounds=args.max_rounds,
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


def _initial_observed_edges(
    track_matrix: np.ndarray,
    *,
    max_gap: int,
    seed_rois: set[int],
    seed_session: int,
) -> set[GapAuditEdge]:
    observed: set[GapAuditEdge] = set()
    for row in _normalize_int_track_matrix(track_matrix):
        if _valid_seed(row, seed_rois, seed_session=seed_session):
            observed.update(_observed_neighbor_edges(row, max_gap=max_gap))
    return observed


def _round_limit(
    track_matrix: np.ndarray,
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    *,
    max_rounds: int | None,
) -> int:
    if max_rounds is not None:
        if int(max_rounds) < 1:
            raise ValueError("max_rounds must be at least 1 when provided")
        return int(max_rounds)
    return max(1, min(len(feature_index), int(np.size(track_matrix))))


def _candidate_confidence(
    candidate: StrictGapCandidate,
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
) -> tuple[float, float, float, float, int, int, int, int]:
    feature = feature_index[candidate.edge]
    session_a, session_b, roi_a, roi_b = candidate.edge
    return (
        float(feature.threshold_margin),
        float(feature.area_ratio),
        min(float(feature.row_margin), float(feature.column_margin)),
        -float(feature.registered_iou),
        -int(session_b) + int(session_a),
        -int(session_a),
        -int(roi_a),
        -int(roi_b),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
