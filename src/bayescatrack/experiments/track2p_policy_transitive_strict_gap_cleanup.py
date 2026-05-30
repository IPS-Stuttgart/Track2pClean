"""Transitive strict gap rescue for Track2p-policy cleanup.

The one-pass strict gap cleanup can accept a high-confidence skip edge and insert
only that target observation.  If another high-confidence skip edge starts from
that newly inserted target, the one-pass candidate scan never sees it because the
source was not present in the baseline component-cleanup matrix.  This module
keeps the same hard gate and conflict checks, but iterates the candidate scan
until no newly inserted observation exposes another accepted strict gap edge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_policy_component_audit import (
    _normalize_int_track_matrix,
)
from bayescatrack.experiments.track2p_policy_gap_edge_audit import (
    GapAuditEdge,
    GapEdgeFeature,
    _observed_neighbor_edges,
)
from bayescatrack.experiments.track2p_policy_strict_gated_gap_cleanup import (
    StrictGapCandidate,
    StrictGapGateConfig,
    _apply_strict_gated_gap_edges_with_report,
    _source_track_id,
    _valid_seed,
    strict_gap_gate_decision,
)


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
