"""Confidence-ranked strict gap cleanup helpers.

The promoted strict-gated gap-cleanup path already keeps gap rescue conservative by
requiring hard local-evidence gates before inserting a direct skip edge. These
helpers keep that gate unchanged but apply accepted candidates in descending
confidence order, so a weaker accepted edge cannot claim a conflicting target ROI
before a stronger accepted edge is tried.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments import (
    track2p_policy_strict_gated_gap_cleanup as strict_gap,
)
from bayescatrack.experiments.track2p_policy_gap_edge_audit import (
    GapEdgeFeature,
    _cell_probability,
)

TRACK2P_POLICY_CONFIDENCE_RANKED_GAP_CLEANUP_METHOD = (
    "track2p-policy-confidence-ranked-gap-cleanup"
)


def rank_strict_gap_candidates(
    candidates: Sequence[strict_gap.StrictGapCandidate],
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[strict_gap.GapAuditEdge, GapEdgeFeature],
) -> tuple[strict_gap.StrictGapCandidate, ...]:
    """Return strict-gap candidates ordered by local merge confidence.

    The existing strict gate decides which edges are admissible. The merge step
    is still conflict-sensitive, because only one track may claim a target ROI in
    a session. Ranking therefore changes only the tie/conflict resolution order
    among already-gated candidates.
    """

    return tuple(
        candidate
        for _index, candidate in sorted(
            enumerate(candidates),
            key=lambda item: _strict_gap_candidate_confidence_key(
                item,
                sessions=sessions,
                feature_index=feature_index,
            ),
            reverse=True,
        )
    )


def apply_confidence_ranked_strict_gap_edges(
    base_tracks: np.ndarray,
    candidates: Sequence[strict_gap.StrictGapCandidate],
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[strict_gap.GapAuditEdge, GapEdgeFeature],
    seed_session: int = 0,
) -> np.ndarray:
    """Insert accepted strict-gap edges after sorting them by confidence."""

    ranked_candidates = rank_strict_gap_candidates(
        candidates,
        sessions=sessions,
        feature_index=feature_index,
    )
    return strict_gap.apply_strict_gated_gap_edges(
        base_tracks,
        ranked_candidates,
        seed_session=seed_session,
    )


def apply_confidence_ranked_strict_gap_edges_with_report(
    base_tracks: np.ndarray,
    candidates: Sequence[strict_gap.StrictGapCandidate],
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[strict_gap.GapAuditEdge, GapEdgeFeature],
    seed_session: int = 0,
) -> tuple[np.ndarray, tuple[strict_gap.StrictGapCandidate, ...], frozenset[int]]:
    """Insert confidence-ranked strict-gap edges and report applied indices.

    Applied indices refer to the returned ranked-candidate tuple, not to the
    original input tuple. This keeps diagnostic rows aligned with the actual
    conflict-resolution order.
    """

    ranked_candidates = rank_strict_gap_candidates(
        candidates,
        sessions=sessions,
        feature_index=feature_index,
    )
    output, applied_indices = strict_gap._apply_strict_gated_gap_edges_with_report(
        base_tracks,
        ranked_candidates,
        seed_session=seed_session,
    )
    return output, ranked_candidates, applied_indices


def _strict_gap_candidate_confidence_key(
    indexed_candidate: tuple[int, strict_gap.StrictGapCandidate],
    *,
    sessions: Sequence[Track2pSession],
    feature_index: Mapping[strict_gap.GapAuditEdge, GapEdgeFeature],
) -> tuple[float, ...]:
    index, candidate = indexed_candidate
    session_a, session_b, roi_a, roi_b = candidate.edge
    feature = feature_index.get(candidate.edge)
    probability_a = _candidate_cell_probability(sessions, session_a, roi_a)
    probability_b = _candidate_cell_probability(sessions, session_b, roi_b)
    min_probability = min(probability_a, probability_b)

    return (
        float(candidate.accepted),
        _finite_or_floor(min_probability),
        _feature_float(feature, "threshold_margin", floor=float("-inf")),
        _feature_float(feature, "row_margin", floor=float("-inf")),
        _feature_float(feature, "column_margin", floor=float("-inf")),
        _feature_float(feature, "area_ratio", floor=float("-inf")),
        -_feature_float(feature, "registered_iou", floor=float("inf")),
        -_feature_float(feature, "centroid_distance", floor=float("inf")),
        -_feature_int(feature, "row_rank"),
        -_feature_int(feature, "column_rank"),
        -float(max(0, int(session_b) - int(session_a))),
        -float(candidate.candidate_track_id),
        -float(index),
    )


def _candidate_cell_probability(
    sessions: Sequence[Track2pSession], session_index: int, roi: int
) -> float:
    if session_index < 0 or session_index >= len(sessions):
        return float("nan")
    return _cell_probability(sessions[int(session_index)], int(roi))


def _feature_float(
    feature: GapEdgeFeature | None,
    name: str,
    *,
    floor: float,
) -> float:
    if feature is None:
        return float(floor)
    return _finite_or_floor(float(getattr(feature, name)), floor=floor)


def _feature_int(feature: GapEdgeFeature | None, name: str) -> float:
    if feature is None:
        return float("-inf")
    value = float(getattr(feature, name))
    return value if np.isfinite(value) else float("-inf")


def _finite_or_floor(value: float, *, floor: float = float("-inf")) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float(floor)
