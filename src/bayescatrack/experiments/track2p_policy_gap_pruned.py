"""Gap-aware conservative pruning for Track2p-policy links.

This module combines two already useful Track2p-policy levers: direct short-gap
rescue and conservative edge pruning. Consecutive links are preferred when they
can support an equally long continuation; a skip link is used when the accepted
one-step link is a dead end and the skip link reaches a longer valid suffix.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
    Track2pPolicyPruneConfig,
    Track2pPolicyPrunedPrediction,
    _roi_indices,
    _thresholded_pruned_hungarian_links,
)

DEFAULT_GAP_PRUNED_MAX_GAP = 2


def emulate_track2p_gap_pruned_tracks(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str = TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    prune_config: Track2pPolicyPruneConfig | None = None,
    max_gap: int = DEFAULT_GAP_PRUNED_MAX_GAP,
) -> Track2pPolicyPrunedPrediction:
    """Return Track2p-policy tracks with pruning and conservative gap rescue."""

    sessions = tuple(sessions)
    max_gap = int(max_gap)
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    if not sessions:
        return Track2pPolicyPrunedPrediction(
            tracks=np.zeros((0, 0), dtype=int), diagnostics=()
        )
    if len(sessions) == 1:
        roi_indices = _roi_indices(sessions[0])
        return Track2pPolicyPrunedPrediction(
            tracks=roi_indices.reshape(-1, 1), diagnostics=()
        )

    prune_config = prune_config or Track2pPolicyPruneConfig()
    links_by_gap, diagnostics = thresholded_pruned_links_by_gap(
        sessions,
        transform_type=transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=prune_config,
        max_gap=max_gap,
    )
    return Track2pPolicyPrunedPrediction(
        tracks=tracks_from_gap_links(sessions, links_by_gap, max_gap=max_gap),
        diagnostics=diagnostics,
    )


def thresholded_pruned_links_by_gap(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    prune_config: Track2pPolicyPruneConfig,
    max_gap: int,
) -> tuple[dict[tuple[int, int], np.ndarray], tuple[Track2pPolicyLinkDiagnostic, ...]]:
    """Build pruned threshold links keyed by ``(source_session, step)``."""

    links_by_gap: dict[tuple[int, int], np.ndarray] = {}
    diagnostics: list[Track2pPolicyLinkDiagnostic] = []
    for session_index in range(len(sessions) - 1):
        for step in range(1, min(int(max_gap), len(sessions) - 1 - session_index) + 1):
            links, pair_diagnostics = _thresholded_pruned_hungarian_links(
                sessions[session_index],
                sessions[session_index + step],
                session_index=session_index,
                transform_type=transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold) * float(step),
                prune_config=prune_config,
            )
            links_by_gap[(session_index, step)] = links
            diagnostics.extend(pair_diagnostics)
    return links_by_gap, tuple(diagnostics)


def tracks_from_gap_links(
    sessions: Sequence[Track2pSession],
    links_by_gap: Mapping[tuple[int, int], np.ndarray],
    *,
    max_gap: int,
) -> np.ndarray:
    """Propagate first-session seeds through lookahead-ranked gap links.

    Candidate links are restricted to the already accepted pruned policy links.
    Among those candidates, propagation chooses the link that reaches the longest
    accepted future suffix. Consecutive links remain the deterministic tie-breaker,
    so the old consecutive-first behavior is preserved whenever it can reach the
    same number of future detections and the same terminal session.
    """

    sessions = tuple(sessions)
    max_gap = int(max_gap)
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    local_tracks = np.full(
        (sessions[0].plane_data.n_rois, len(sessions)), -1, dtype=int
    )
    seed_rois = seed_rois_with_outgoing_gap_links(
        links_by_gap,
        max_gap=max_gap,
        first_session_size=sessions[0].plane_data.n_rois,
    )
    if seed_rois.size:
        local_tracks[seed_rois, 0] = seed_rois

    suffix_memo: dict[tuple[int, int], tuple[int, int]] = {}
    for row_index in range(local_tracks.shape[0]):
        current_session = 0
        current = int(local_tracks[row_index, current_session])
        if current < 0:
            continue
        while current_session < len(sessions) - 1:
            candidate = _best_gap_link_candidate(
                current_session,
                current,
                links_by_gap,
                max_gap=max_gap,
                n_sessions=len(sessions),
                suffix_memo=suffix_memo,
            )
            if candidate is None:
                break
            step, current = candidate
            current_session += step
            local_tracks[row_index, current_session] = current

    suite2p_tracks = np.full_like(local_tracks, -1)
    for session_index, roi_indices in enumerate(
        _roi_indices(session) for session in sessions
    ):
        valid = local_tracks[:, session_index] >= 0
        if np.any(valid):
            suite2p_tracks[valid, session_index] = roi_indices[
                local_tracks[valid, session_index]
            ]
    return suite2p_tracks


def seed_rois_with_outgoing_gap_links(
    links_by_gap: Mapping[tuple[int, int], np.ndarray],
    *,
    max_gap: int,
    first_session_size: int,
) -> np.ndarray:
    """Return first-session ROI ids that can start a propagated track."""

    seeds: set[int] = set()
    for step in range(1, int(max_gap) + 1):
        links = links_by_gap.get((0, step))
        if links is None or not links.size:
            continue
        for source_roi in links[:, 0]:
            if 0 <= int(source_roi) < int(first_session_size):
                seeds.add(int(source_roi))
    return np.asarray(sorted(seeds), dtype=int)


def _best_gap_link_candidate(
    source_session: int,
    source_roi: int,
    links_by_gap: Mapping[tuple[int, int], np.ndarray],
    *,
    max_gap: int,
    n_sessions: int,
    suffix_memo: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int] | None:
    """Return the next accepted gap link that maximizes reachable suffix length."""

    best_candidate: tuple[int, int] | None = None
    best_score: tuple[int, int, int, int] | None = None
    for step, target_roi in _gap_link_candidates(
        source_session,
        source_roi,
        links_by_gap,
        max_gap=max_gap,
        n_sessions=n_sessions,
    ):
        target_session = int(source_session) + int(step)
        future_count, terminal_session = _best_suffix_score(
            target_session,
            target_roi,
            links_by_gap,
            max_gap=max_gap,
            n_sessions=n_sessions,
            suffix_memo=suffix_memo,
        )
        score = (
            1 + int(future_count),
            int(terminal_session),
            -int(step),
            -int(target_roi),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_candidate = (int(step), int(target_roi))
    return best_candidate


def _best_suffix_score(
    source_session: int,
    source_roi: int,
    links_by_gap: Mapping[tuple[int, int], np.ndarray],
    *,
    max_gap: int,
    n_sessions: int,
    suffix_memo: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    """Return ``(future_detection_count, terminal_session)`` from one node."""

    key = (int(source_session), int(source_roi))
    if key in suffix_memo:
        return suffix_memo[key]

    best_score = (0, int(source_session), 0, 0)
    for step, target_roi in _gap_link_candidates(
        source_session,
        source_roi,
        links_by_gap,
        max_gap=max_gap,
        n_sessions=n_sessions,
    ):
        target_session = int(source_session) + int(step)
        future_count, terminal_session = _best_suffix_score(
            target_session,
            target_roi,
            links_by_gap,
            max_gap=max_gap,
            n_sessions=n_sessions,
            suffix_memo=suffix_memo,
        )
        score = (
            1 + int(future_count),
            int(terminal_session),
            -int(step),
            -int(target_roi),
        )
        if score > best_score:
            best_score = score

    suffix_memo[key] = (int(best_score[0]), int(best_score[1]))
    return suffix_memo[key]


def _gap_link_candidates(
    source_session: int,
    source_roi: int,
    links_by_gap: Mapping[tuple[int, int], np.ndarray],
    *,
    max_gap: int,
    n_sessions: int,
) -> Iterator[tuple[int, int]]:
    """Yield accepted ``(step, target_roi)`` links from one local detection."""

    remaining_sessions = int(n_sessions) - 1 - int(source_session)
    for step in range(1, min(int(max_gap), remaining_sessions) + 1):
        links = _as_link_matrix(links_by_gap.get((int(source_session), int(step))))
        if links.size == 0:
            continue
        matches = links[links[:, 0] == int(source_roi), 1]
        for target_roi in matches:
            yield int(step), int(target_roi)


def _as_link_matrix(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros((0, 2), dtype=int)
    links = np.asarray(value, dtype=int)
    if links.size == 0:
        return np.zeros((0, 2), dtype=int)
    if links.ndim != 2 or links.shape[1] != 2:
        raise ValueError(f"gap links must have shape (n, 2), got {links.shape}")
    return links
