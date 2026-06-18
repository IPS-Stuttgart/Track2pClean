"""Strictly gated gap rescue after Track2p-policy component cleanup.

This row preserves the promoted adjacent-only component-cleanup prediction as
the baseline.  Gap rescue is used only as a candidate generator: direct skip
edges absent from the component-cleanup matrix are accepted only if they pass a
hard feature gate, then their candidate suffix is merged into the baseline when
that merge does not create ROI-observation conflicts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
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
    _accepted_pair_features,
    _cell_probability,
    _component_cleanup_prediction,
    _feature_value,
    _observed_neighbor_edges,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import _roi_indices

TRACK2P_POLICY_STRICT_GATED_GAP_CLEANUP_METHOD = (
    "track2p-policy-strict-gated-gap-cleanup"
)
STRICT_GATED_GAP_DEFAULT_MAX_GAP = 2


@dataclass(frozen=True)
class StrictGapGateConfig:
    """Hard feature gate for post-component-cleanup gap rescue candidates."""

    gap_length: int = 2
    min_area_ratio: float = 0.90
    min_cell_probability: float = 0.80
    max_registered_iou: float = 0.55
    min_row_margin: float = 0.0
    min_column_margin: float = 0.0
    min_threshold_margin: float = 0.20

    def __post_init__(self) -> None:
        _require_positive_int(self.gap_length, name="gap_length")
        _require_probability_like(self.min_area_ratio, name="min_area_ratio")
        _require_probability_like(
            self.min_cell_probability, name="min_cell_probability"
        )
        _require_nonnegative(self.max_registered_iou, name="max_registered_iou")
        _require_finite(self.min_row_margin, name="min_row_margin")
        _require_finite(self.min_column_margin, name="min_column_margin")
        _require_finite(self.min_threshold_margin, name="min_threshold_margin")


@dataclass(frozen=True)
class StrictGapCandidate:
    """One gap-rescue candidate edge and the row that proposed it."""

    edge: GapAuditEdge
    candidate_track_id: int
    accepted: bool
    reason: str


def run_track2p_policy_strict_gated_gap_cleanup(
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
    """Run component cleanup, then merge only strictly gated gap candidates."""

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
                "Track2p-policy strict gated gap cleanup requires independent "
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
        candidates = strict_gated_gap_edge_candidates(
            base_full,
            sessions=sessions,
            feature_index=feature_index,
            gate_config=gate_config,
            seed_rois=_seed_rois(
                reference_tracks, seed_session=policy_config.seed_session
            ),
            seed_session=policy_config.seed_session,
        )
        cleaned, applied_candidate_indices = _apply_strict_gated_gap_edges_with_report(
            base_full,
            candidates,
            seed_session=policy_config.seed_session,
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        gate_accepted_edges = int(sum(candidate.accepted for candidate in candidates))
        applied_edges = int(len(applied_candidate_indices))
        scores = {
            **scores,
            "track2p_policy_variant": TRACK2P_POLICY_STRICT_GATED_GAP_CLEANUP_METHOD,
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
                variant="Track2p-policy component cleanup + strict gated gap rescue",
                method=cast(Any, TRACK2P_POLICY_STRICT_GATED_GAP_CLEANUP_METHOD),
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
                cell_probability_threshold=float(
                    policy_config.cell_probability_threshold
                ),
                transform_type=policy_config.transform_type,
                max_gap=int(policy_config.max_gap),
                applied_candidate_indices=applied_candidate_indices,
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(edge_rows))


def strict_gated_gap_edge_candidates(
    base_tracks: np.ndarray,
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
    seed_rois: set[int],
    seed_session: int = 0,
) -> tuple[StrictGapCandidate, ...]:
    """Return direct gap-edge candidates absent from component cleanup."""

    base_matrix = _normalize_int_track_matrix(base_tracks)
    base_counts: Counter[GapAuditEdge] = Counter()
    for row in base_matrix:
        if _valid_seed(row, seed_rois, seed_session=seed_session):
            base_counts.update(
                _observed_neighbor_edges(row, max_gap=int(gate_config.gap_length))
            )

    decisions: list[StrictGapCandidate] = []
    seen_edges: set[GapAuditEdge] = set()
    for edge in sorted(feature_index):
        if edge in seen_edges or base_counts.get(edge, 0) > 0:
            continue
        source_track_id = _source_track_id(
            base_matrix,
            edge,
            seed_rois=seed_rois,
            seed_session=seed_session,
        )
        if source_track_id is None:
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
                candidate_track_id=int(source_track_id),
                accepted=accepted,
                reason=reason,
            )
        )
        seen_edges.add(edge)
    return tuple(decisions)


def strict_gated_gap_candidates(
    base_tracks: np.ndarray,
    candidate_tracks: np.ndarray,
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
    max_gap: int,
    seed_rois: set[int],
    seed_session: int = 0,
) -> tuple[StrictGapCandidate, ...]:
    """Return candidate delta edges with gate decisions."""

    return _evaluate_strict_gap_candidates(
        _delta_gap_candidate_occurrences(
            base_tracks,
            candidate_tracks,
            max_gap=max_gap,
            seed_rois=seed_rois,
            seed_session=seed_session,
        ),
        sessions=sessions,
        feature_index=feature_index,
        gate_config=gate_config,
    )


def _delta_gap_candidate_occurrences(
    base_tracks: np.ndarray,
    candidate_tracks: np.ndarray,
    *,
    max_gap: int,
    seed_rois: set[int],
    seed_session: int,
) -> tuple[tuple[GapAuditEdge, int], ...]:
    """Return candidate edge occurrences absent from the baseline prediction."""

    base_counts: Counter[GapAuditEdge] = Counter()
    for row in _normalize_int_track_matrix(base_tracks):
        if _valid_seed(row, seed_rois, seed_session=seed_session):
            base_counts.update(_observed_neighbor_edges(row, max_gap=max_gap))
    candidate_counts: Counter[GapAuditEdge] = Counter()
    occurrences: list[tuple[GapAuditEdge, int]] = []
    for track_id, row in enumerate(_normalize_int_track_matrix(candidate_tracks)):
        if not _valid_seed(row, seed_rois, seed_session=seed_session):
            continue
        for edge in _observed_neighbor_edges(row, max_gap=max_gap):
            candidate_counts[edge] += 1
            if candidate_counts[edge] <= base_counts.get(edge, 0):
                continue
            occurrences.append((edge, int(track_id)))
    return tuple(occurrences)


def _evaluate_strict_gap_candidates(
    occurrences: Sequence[tuple[GapAuditEdge, int]],
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    gate_config: StrictGapGateConfig,
) -> tuple[StrictGapCandidate, ...]:
    """Apply the strict gate to pre-enumerated delta edge occurrences."""

    decisions: list[StrictGapCandidate] = []
    for edge, track_id in occurrences:
        feature = feature_index.get(edge)
        accepted, reason = strict_gap_gate_decision(
            edge,
            feature,
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
    return tuple(decisions)


def strict_gap_feature_index_for_gap_length(
    sessions: Sequence[Track2pSession],
    *,
    gap_length: int,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> dict[GapAuditEdge, GapEdgeFeature]:
    """Compute accepted registered-IoU gap-link features for one gap length."""

    sessions = tuple(sessions)
    step = int(gap_length)
    if step < 1:
        raise ValueError("gap_length must be at least 1")
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    output: dict[GapAuditEdge, GapEdgeFeature] = {}
    for session_a in range(max(0, len(sessions) - step)):
        session_b = session_a + step
        local_features = _accepted_pair_features(
            sessions[session_a],
            sessions[session_b],
            transform_type=transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold) * float(step),
        )
        source_indices = roi_indices_by_session[session_a]
        target_indices = roi_indices_by_session[session_b]
        for (local_a, local_b), feature in local_features.items():
            output[
                (
                    int(session_a),
                    int(session_b),
                    int(source_indices[local_a]),
                    int(target_indices[local_b]),
                )
            ] = feature
    return output


def strict_gap_feature_subset(
    sessions: Sequence[Track2pSession],
    *,
    edges: set[GapAuditEdge] | frozenset[GapAuditEdge],
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> dict[GapAuditEdge, GapEdgeFeature]:
    """Compute registered-IoU features only for requested candidate edges."""

    if not edges:
        return {}
    sessions = tuple(sessions)
    edges_by_pair: dict[tuple[int, int], set[GapAuditEdge]] = {}
    for edge in edges:
        session_a, session_b, _roi_a, _roi_b = edge
        edges_by_pair.setdefault((int(session_a), int(session_b)), set()).add(edge)

    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    output: dict[GapAuditEdge, GapEdgeFeature] = {}
    for (session_a, session_b), requested_edges in edges_by_pair.items():
        if not (0 <= session_a < session_b < len(sessions)):
            continue
        step = int(session_b) - int(session_a)
        local_features = _accepted_pair_features(
            sessions[session_a],
            sessions[session_b],
            transform_type=transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold) * float(step),
        )
        source_indices = roi_indices_by_session[session_a]
        target_indices = roi_indices_by_session[session_b]
        for (local_a, local_b), feature in local_features.items():
            edge = (
                int(session_a),
                int(session_b),
                int(source_indices[local_a]),
                int(target_indices[local_b]),
            )
            if edge in requested_edges:
                output[edge] = feature
    return output


def _source_track_id(
    track_matrix: np.ndarray,
    edge: GapAuditEdge,
    *,
    seed_rois: set[int],
    seed_session: int,
) -> int | None:
    session_a, _session_b, roi_a, _roi_b = edge
    for track_id, row in enumerate(_normalize_int_track_matrix(track_matrix)):
        if not _valid_seed(row, seed_rois, seed_session=seed_session):
            continue
        if int(row[session_a]) == int(roi_a):
            return int(track_id)
    return None


def strict_gap_gate_decision(
    edge: GapAuditEdge,
    feature: GapEdgeFeature | None,
    *,
    sessions: Sequence[Track2pSession],
    gate_config: StrictGapGateConfig,
) -> tuple[bool, str]:
    """Return whether one delta edge passes the strict gap-rescue gate."""

    session_a, session_b, roi_a, roi_b = edge
    if int(session_b) - int(session_a) != int(gate_config.gap_length):
        return False, "gap-length"
    if feature is None:
        return False, "missing-feature"
    probability_a = _cell_probability(sessions[session_a], int(roi_a))
    probability_b = _cell_probability(sessions[session_b], int(roi_b))
    checks = (
        (feature.area_ratio >= gate_config.min_area_ratio, "area-ratio"),
        (
            min(probability_a, probability_b) >= gate_config.min_cell_probability,
            "cell-probability",
        ),
        (feature.registered_iou <= gate_config.max_registered_iou, "registered-iou"),
        (feature.row_margin >= gate_config.min_row_margin, "row-margin"),
        (feature.column_margin >= gate_config.min_column_margin, "column-margin"),
        (
            feature.threshold_margin >= gate_config.min_threshold_margin,
            "threshold-margin",
        ),
    )
    failed = [reason for passed, reason in checks if not passed]
    return (not failed, "accepted" if not failed else ";".join(failed))


def apply_strict_gated_gap_candidates(
    base_tracks: np.ndarray,
    candidate_tracks: np.ndarray,
    candidates: Sequence[StrictGapCandidate],
    *,
    seed_session: int = 0,
) -> np.ndarray:
    """Merge accepted candidate suffixes into component-cleanup tracks."""

    output, _ = _apply_strict_gated_gap_candidates_with_report(
        base_tracks,
        candidate_tracks,
        candidates,
        seed_session=seed_session,
    )
    return output


def apply_strict_gated_gap_edges(
    base_tracks: np.ndarray,
    candidates: Sequence[StrictGapCandidate],
    *,
    seed_session: int = 0,
) -> np.ndarray:
    """Insert accepted strict gap targets into component-cleanup tracks."""

    output, _ = _apply_strict_gated_gap_edges_with_report(
        base_tracks,
        candidates,
        seed_session=seed_session,
    )
    return output


def _apply_strict_gated_gap_edges_with_report(
    base_tracks: np.ndarray,
    candidates: Sequence[StrictGapCandidate],
    *,
    seed_session: int = 0,
) -> tuple[np.ndarray, frozenset[int]]:
    """Insert accepted strict gap targets and report applied candidate indices."""

    output = _normalize_int_track_matrix(base_tracks).copy()
    observation_counts = _observation_counter(output)
    applied: set[int] = set()
    for candidate_index, candidate in enumerate(candidates):
        if not candidate.accepted:
            continue
        updated = _merge_target_edge(
            output,
            candidate.edge,
            track_id=int(candidate.candidate_track_id),
            observation_counts=observation_counts,
            seed_session=seed_session,
        )
        if updated is not output:
            applied.add(int(candidate_index))
        output = updated
    return output, frozenset(applied)


def _apply_strict_gated_gap_candidates_with_report(
    base_tracks: np.ndarray,
    candidate_tracks: np.ndarray,
    candidates: Sequence[StrictGapCandidate],
    *,
    seed_session: int = 0,
) -> tuple[np.ndarray, frozenset[int]]:
    """Merge accepted candidates and report which candidate indices applied."""

    output = _normalize_int_track_matrix(base_tracks).copy()
    candidate_matrix = _normalize_int_track_matrix(candidate_tracks)
    observation_counts = _observation_counter(output)
    applied: set[int] = set()
    for candidate_index, candidate in enumerate(candidates):
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


def _merge_target_edge(
    base_tracks: np.ndarray,
    edge: GapAuditEdge,
    *,
    track_id: int,
    observation_counts: Counter[tuple[int, int]],
    seed_session: int,
) -> np.ndarray:
    session_a, session_b, roi_a, roi_b = edge
    if not 0 <= int(track_id) < base_tracks.shape[0]:
        return base_tracks
    base_row = base_tracks[int(track_id)].copy()
    if not _valid_seed_value(base_row, seed_session=seed_session):
        return base_tracks
    if int(base_row[session_a]) != int(roi_a):
        return base_tracks
    if int(base_row[session_b]) == int(roi_b):
        return base_tracks
    if int(base_row[session_b]) >= 0:
        return base_tracks
    if observation_counts.get((int(session_b), int(roi_b)), 0) > 0:
        return base_tracks

    base_row[session_b] = int(roi_b)
    observation_counts[(int(session_b), int(roi_b))] += 1
    output = base_tracks.copy()
    output[int(track_id)] = base_row
    return output


def _merge_candidate_row(
    base_tracks: np.ndarray,
    candidate_row: np.ndarray,
    edge: GapAuditEdge,
    *,
    observation_counts: Counter[tuple[int, int]],
    seed_session: int,
) -> np.ndarray:
    session_a, session_b, roi_a, roi_b = edge
    if candidate_row[session_a] != int(roi_a) or candidate_row[session_b] != int(roi_b):
        return base_tracks
    if not _valid_seed_value(candidate_row, seed_session=seed_session):
        return base_tracks

    seed_roi = int(candidate_row[seed_session])
    matches = np.flatnonzero(base_tracks[:, seed_session] == seed_roi)
    if matches.size == 0:
        return _append_candidate_row(
            base_tracks,
            candidate_row,
            observation_counts=observation_counts,
        )

    row_index = int(matches[0])
    base_row = base_tracks[row_index].copy()
    if base_row[session_a] != int(roi_a):
        return base_tracks

    additions: list[tuple[int, int]] = []
    for session_index in range(int(session_b), candidate_row.size):
        roi = int(candidate_row[session_index])
        if roi < 0:
            continue
        if base_row[session_index] >= 0 and int(base_row[session_index]) != roi:
            return base_tracks
        if base_row[session_index] < 0:
            if observation_counts.get((session_index, roi), 0) > 0:
                return base_tracks
            additions.append((session_index, roi))
    if not additions:
        return base_tracks
    for session_index, roi in additions:
        base_row[session_index] = roi
        observation_counts[(session_index, roi)] += 1
    output = base_tracks.copy()
    output[row_index] = base_row
    return output


def _append_candidate_row(
    base_tracks: np.ndarray,
    candidate_row: np.ndarray,
    *,
    observation_counts: Counter[tuple[int, int]],
) -> np.ndarray:
    observations = [
        (session_index, int(roi))
        for session_index, roi in enumerate(candidate_row)
        if int(roi) >= 0
    ]
    if any(observation_counts.get(observation, 0) > 0 for observation in observations):
        return base_tracks
    for observation in observations:
        observation_counts[observation] += 1
    return np.vstack([base_tracks, candidate_row.astype(int, copy=True)])


def _candidate_rows(
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
    rows: list[dict[str, float | int | str]] = []
    applied_indices = set(applied_candidate_indices)
    for candidate_index, candidate in enumerate(candidates):
        session_a, session_b, roi_a, roi_b = candidate.edge
        feature = feature_index.get(candidate.edge)
        rows.append(
            {
                "subject": subject,
                "session_a": int(session_a),
                "session_b": int(session_b),
                "session_a_name": str(sessions[session_a].session_name),
                "session_b_name": str(sessions[session_b].session_name),
                "roi_a": int(roi_a),
                "roi_b": int(roi_b),
                "gap_length": int(session_b - session_a),
                "candidate_track_id": int(candidate.candidate_track_id),
                "accepted": int(candidate.accepted),
                "applied": int(candidate_index in applied_indices),
                "reject_reason": str(candidate.reason),
                "registered_iou": _feature_value(feature, "registered_iou"),
                "threshold_margin": _feature_value(feature, "threshold_margin"),
                "row_margin": _feature_value(feature, "row_margin"),
                "column_margin": _feature_value(feature, "column_margin"),
                "centroid_distance": _feature_value(feature, "centroid_distance"),
                "area_ratio": _feature_value(feature, "area_ratio"),
                "cell_probability_a": _cell_probability(
                    sessions[session_a], int(roi_a)
                ),
                "cell_probability_b": _cell_probability(
                    sessions[session_b], int(roi_b)
                ),
                "gate_gap_length": int(gate_config.gap_length),
                "gate_min_area_ratio": float(gate_config.min_area_ratio),
                "gate_min_cell_probability": float(gate_config.min_cell_probability),
                "gate_max_registered_iou": float(gate_config.max_registered_iou),
                "gate_min_row_margin": float(gate_config.min_row_margin),
                "gate_min_column_margin": float(gate_config.min_column_margin),
                "gate_min_threshold_margin": float(gate_config.min_threshold_margin),
                "threshold_method": str(threshold_method),
                "iou_distance_threshold": float(iou_distance_threshold),
                "cell_probability_threshold": float(cell_probability_threshold),
                "transform_type": str(transform_type),
                "max_gap": int(max_gap),
            }
        )
    return rows


def write_strict_gap_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write strict gated gap diagnostics as CSV or JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for strict gated gap cleanup."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-strict-gated-gap-cleanup",
        description=(
            "Run Track2p-policy component cleanup, then admit only hard-gated "
            "gap-rescue candidate edges."
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
    """Run the strict gated gap cleanup CLI."""

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
    output = run_track2p_policy_strict_gated_gap_cleanup(
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


def _seed_rois(track_matrix: np.ndarray, *, seed_session: int) -> set[int]:
    matrix = _normalize_int_track_matrix(track_matrix)
    if seed_session < 0 or seed_session >= matrix.shape[1]:
        return set()
    return {int(row[seed_session]) for row in matrix if row[seed_session] >= 0}


def _valid_seed(row: np.ndarray, seed_rois: set[int], *, seed_session: int) -> bool:
    return bool(
        0 <= seed_session < row.size
        and row[seed_session] >= 0
        and int(row[seed_session]) in seed_rois
    )


def _valid_seed_value(row: np.ndarray, *, seed_session: int) -> bool:
    return bool(0 <= seed_session < row.size and row[seed_session] >= 0)


def _observation_counter(track_matrix: np.ndarray) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for row in _normalize_int_track_matrix(track_matrix):
        for session_index, roi in enumerate(row):
            if roi >= 0:
                counts[(int(session_index), int(roi))] += 1
    return counts


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be at least 1")


def _finite_float_value(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _require_probability_like(value: float, *, name: str) -> None:
    numeric = _finite_float_value(value, name=name)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_nonnegative(value: float, *, name: str) -> None:
    if _finite_float_value(value, name=name) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_finite(value: float, *, name: str) -> None:
    _finite_float_value(value, name=name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
