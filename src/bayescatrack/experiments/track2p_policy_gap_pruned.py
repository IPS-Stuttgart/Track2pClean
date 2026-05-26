"""Gap-aware conservative pruning for Track2p-policy links.

This module combines two already useful Track2p-policy levers: direct short-gap
rescue and conservative edge pruning. Consecutive links are always preferred;
a skip link is used only when no accepted adjacent continuation exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
    """Propagate first-session seeds through consecutive-first gap links."""

    sessions = tuple(sessions)
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

    for row_index in range(local_tracks.shape[0]):
        current_session = 0
        current = int(local_tracks[row_index, current_session])
        if current < 0:
            continue
        while current_session < len(sessions) - 1:
            matched = False
            for step in range(
                1, min(int(max_gap), len(sessions) - 1 - current_session) + 1
            ):
                links = links_by_gap.get((current_session, step))
                if links is None or not links.size:
                    continue
                matches = np.flatnonzero(links[:, 0] == current)
                if matches.size == 0:
                    continue
                current_session += step
                current = int(links[matches[0], 1])
                local_tracks[row_index, current_session] = current
                matched = True
                break
            if not matched:
                break

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
