"""Opt-in growth-history prediction scoring for FullMHT.

The base FullMHT runner scores each candidate edge from scan-pair diagnostics.
This module adds a history-conditioned dynamics term: when a track already has a
selected edge history, a new candidate is penalized if its label-free growth-field
residual, Mahalanobis residual, local deformation, or IoU diagnostics deteriorate
sharply relative to that same identity history.

This is deliberately not a post-hoc veto.  The penalty is applied while the scan
assignment cost matrix is built, before Murty scan hypotheses are generated.  It
therefore changes which full identity histories survive beam pruning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bayescatrack.experiments.full_mht_scan_history_dynamics_integration import (
    ScanHistoryEdgeFeatures,
    parse_selected_edge_summary,
)

_REGISTERED_IOU_DROP = 0.15
_SHIFTED_IOU_DROP = 0.20
_GROWTH_RESIDUAL_OFFSET = 1.50
_GROWTH_MAHALANOBIS_OFFSET = 2.00
_LOCAL_DEFORMATION_OFFSET = 0.30
_CONTEXT_STACK: list[GrowthHistoryPredictionContext] = []


@dataclass(frozen=True)
class GrowthHistoryPredictionContext:
    """The partial hypothesis currently being expanded by FullMHT."""

    hypothesis: Any
    session_index: int


def install_full_mht_growth_history_prediction_scoring() -> None:
    """Install history-conditioned growth prediction into FullMHT scoring."""

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    if getattr(full_mht, "_bayescatrack_growth_history_prediction_scoring", False):
        return

    original_expand = full_mht._expand_hypothesis_scan
    original_edge_score = full_mht._edge_score
    original_selected_edge_summary = full_mht._selected_edge_summary

    def _expand_hypothesis_scan_with_growth_history(
        hypothesis: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        session_index = _session_index_from_call(args, kwargs)
        _CONTEXT_STACK.append(
            GrowthHistoryPredictionContext(
                hypothesis=hypothesis,
                session_index=int(session_index),
            )
        )
        try:
            return original_expand(hypothesis, *args, **kwargs)
        finally:
            _CONTEXT_STACK.pop()

    def _edge_score_with_growth_history(
        sessions: Sequence[Any],
        matrices: Any,
        *,
        target_session: int,
        source_local: int,
        target_local: int,
        config: Any,
        track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    ) -> float:
        score = original_edge_score(
            sessions,
            matrices,
            target_session=target_session,
            source_local=source_local,
            target_local=target_local,
            config=config,
            track2p_prior_edges=track2p_prior_edges,
        )
        weight = _growth_history_prediction_weight(config)
        if weight <= 0.0:
            return float(score)
        penalty = growth_history_prediction_penalty_for_candidate(
            matrices,
            source_local=int(source_local),
            target_local=int(target_local),
            target_session=int(target_session),
            config=config,
        )
        return float(score) - weight * penalty

    def _selected_edge_summary_with_growth_history(
        sessions: Sequence[Any],
        matrices: Any,
        *,
        active_source: Any,
        target_session: int,
        target_roi: int,
        config: Any,
        track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    ) -> dict[str, Any]:
        output = original_selected_edge_summary(
            sessions,
            matrices,
            active_source=active_source,
            target_session=target_session,
            target_roi=target_roi,
            config=config,
            track2p_prior_edges=track2p_prior_edges,
        )
        weight = _growth_history_prediction_weight(config)
        if weight <= 0.0:
            return output
        source_matches = np.flatnonzero(
            np.asarray(matrices.source_indices, dtype=int)
            == int(active_source.source_roi)
        )
        target_matches = np.flatnonzero(
            np.asarray(matrices.target_indices, dtype=int) == int(target_roi)
        )
        if source_matches.size == 0 or target_matches.size == 0:
            return output
        penalty = growth_history_prediction_penalty_for_candidate(
            matrices,
            source_local=int(source_matches[0]),
            target_local=int(target_matches[0]),
            target_session=int(target_session),
            config=config,
        )
        weighted = weight * float(penalty)
        output["growth_history_prediction_penalty"] = float(penalty)
        output["growth_history_prediction_weighted_penalty"] = float(weighted)
        output["summary"] = (
            f'{output["summary"]}'
            f"|growth_pred={full_mht._diagnostic_float(float(penalty))}"
            f"|growth_pred_weighted={full_mht._diagnostic_float(float(weighted))}"
        )
        return output

    full_mht._expand_hypothesis_scan = _expand_hypothesis_scan_with_growth_history
    full_mht._edge_score = _edge_score_with_growth_history
    full_mht._selected_edge_summary = _selected_edge_summary_with_growth_history
    full_mht._bayescatrack_growth_history_prediction_original_expand = original_expand
    full_mht._bayescatrack_growth_history_prediction_original_edge_score = original_edge_score
    full_mht._bayescatrack_growth_history_prediction_original_selected_edge_summary = (
        original_selected_edge_summary
    )
    full_mht._bayescatrack_growth_history_prediction_scoring = True


def growth_history_prediction_penalty_for_candidate(
    matrices: Any,
    *,
    source_local: int,
    target_local: int,
    target_session: int,
    config: Any,
) -> float:
    """Return the dynamics penalty for one candidate edge in the active context."""

    if not _CONTEXT_STACK:
        return 0.0
    context = _CONTEXT_STACK[-1]
    candidate = _candidate_features(
        matrices,
        source_local=int(source_local),
        target_local=int(target_local),
        target_session=int(target_session),
    )
    previous = _previous_row_features_for_candidate(
        context.hypothesis,
        candidate=candidate,
        config=config,
    )
    min_edges = _growth_history_prediction_min_edges(config)
    if len(previous) < min_edges:
        return 0.0
    penalty = row_growth_history_prediction_penalty(previous, candidate)
    scale = _growth_history_prediction_scale(config)
    if scale > 0.0:
        penalty /= scale
    clip = _growth_history_prediction_clip(config)
    if clip > 0.0:
        penalty = min(float(clip), float(penalty))
    return float(max(0.0, penalty))


def row_growth_history_prediction_penalty(
    previous: Sequence[ScanHistoryEdgeFeatures],
    candidate: ScanHistoryEdgeFeatures,
) -> float:
    """Penalize a candidate that is inconsistent with a row's edge history."""

    if not previous:
        return 0.0
    registered = _feature_array(previous, "registered_iou")
    shifted = _feature_array(previous, "shifted_iou")
    growth = _feature_array(previous, "growth_residual")
    mahalanobis = _feature_array(previous, "growth_mahalanobis")
    local = _feature_array(previous, "local_deformation")
    penalty = 0.0
    penalty += _low_prediction_penalty(
        registered,
        float(candidate.registered_iou),
        allowed_drop=_REGISTERED_IOU_DROP,
    )
    penalty += _low_prediction_penalty(
        shifted,
        float(candidate.shifted_iou),
        allowed_drop=_SHIFTED_IOU_DROP,
    )
    penalty += _high_prediction_penalty(
        growth,
        float(candidate.growth_residual),
        allowed_offset=_GROWTH_RESIDUAL_OFFSET,
    )
    penalty += _high_prediction_penalty(
        mahalanobis,
        float(candidate.growth_mahalanobis),
        allowed_offset=_GROWTH_MAHALANOBIS_OFFSET,
    )
    penalty += _high_prediction_penalty(
        local,
        float(candidate.local_deformation),
        allowed_offset=_LOCAL_DEFORMATION_OFFSET,
    )
    return float(max(0.0, penalty))


def _previous_row_features_for_candidate(
    hypothesis: Any,
    *,
    candidate: ScanHistoryEdgeFeatures,
    config: Any,
) -> tuple[ScanHistoryEdgeFeatures, ...]:
    tracks = np.asarray(getattr(hypothesis, "tracks", hypothesis), dtype=int)
    if tracks.ndim != 2 or tracks.size == 0:
        return tuple()
    row_index = _active_row_for_candidate(tracks, candidate=candidate, config=config)
    if row_index is None:
        return tuple()
    row = np.asarray(tracks[int(row_index)], dtype=int)
    observed = np.flatnonzero(row[: int(candidate.edge[1])] >= 0)
    if observed.size < 2:
        return tuple()
    feature_map = _selected_edge_feature_map(getattr(hypothesis, "history", ()))
    features: list[ScanHistoryEdgeFeatures] = []
    for left, right in zip(observed[:-1], observed[1:]):
        edge = (int(left), int(right), int(row[int(left)]), int(row[int(right)]))
        parsed = feature_map.get(edge)
        if parsed is not None:
            features.append(parsed)
    return tuple(features)


def _active_row_for_candidate(
    tracks: np.ndarray,
    *,
    candidate: ScanHistoryEdgeFeatures,
    config: Any,
) -> int | None:
    source_session, target_session, source_roi, _target_roi = candidate.edge
    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    active_sources = full_mht._active_track_sources(
        tracks,
        session_index=int(target_session) - 1,
        max_gap=int(getattr(config, "max_gap", 0)),
    )
    matches = [
        int(active.row_index)
        for active in active_sources
        if int(active.source_session) == int(source_session)
        and int(active.source_roi) == int(source_roi)
    ]
    if not matches:
        return None
    return int(matches[0])


def _candidate_features(
    matrices: Any,
    *,
    source_local: int,
    target_local: int,
    target_session: int,
) -> ScanHistoryEdgeFeatures:
    source_roi = int(np.asarray(matrices.source_indices, dtype=int)[int(source_local)])
    target_roi = int(np.asarray(matrices.target_indices, dtype=int)[int(target_local)])
    return ScanHistoryEdgeFeatures(
        edge=(
            int(matrices.source_session),
            int(target_session),
            source_roi,
            target_roi,
        ),
        registered_iou=_finite_matrix_value(matrices.registered_iou, source_local, target_local),
        shifted_iou=_finite_matrix_value(matrices.shifted_iou, source_local, target_local),
        growth_residual=_finite_matrix_value(matrices.growth_residual, source_local, target_local),
        growth_mahalanobis=_finite_matrix_value(matrices.growth_mahalanobis, source_local, target_local),
        local_deformation=_finite_matrix_value(matrices.local_deformation, source_local, target_local),
    )


def _selected_edge_feature_map(
    history: Sequence[dict[str, Any]],
) -> dict[tuple[int, int, int, int], ScanHistoryEdgeFeatures]:
    output: dict[tuple[int, int, int, int], ScanHistoryEdgeFeatures] = {}
    for scan in history:
        raw = str(scan.get("selected_edge_summaries", ""))
        if not raw:
            continue
        for item in raw.split(";"):
            parsed = parse_selected_edge_summary(item)
            if parsed is not None:
                output[parsed.edge] = parsed
    return output


def _feature_array(features: Sequence[ScanHistoryEdgeFeatures], name: str) -> np.ndarray:
    return np.asarray([float(getattr(feature, name)) for feature in features], dtype=float)


def _low_prediction_penalty(
    previous: np.ndarray,
    candidate: float,
    *,
    allowed_drop: float,
) -> float:
    finite = np.asarray(previous, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or not np.isfinite(float(candidate)):
        return 0.0
    predicted = float(np.median(finite))
    return float(max(0.0, predicted - float(candidate) - float(allowed_drop)))


def _high_prediction_penalty(
    previous: np.ndarray,
    candidate: float,
    *,
    allowed_offset: float,
) -> float:
    finite = np.asarray(previous, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or not np.isfinite(float(candidate)):
        return 0.0
    predicted = float(np.median(finite))
    return float(max(0.0, float(candidate) - predicted - float(allowed_offset)))


def _finite_matrix_value(matrix: Any, source_local: int, target_local: int) -> float:
    try:
        value = float(np.asarray(matrix, dtype=float)[int(source_local), int(target_local)])
    except (IndexError, TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _session_index_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    if "session_index" in kwargs:
        return int(kwargs["session_index"])
    if len(args) >= 3:
        return int(args[2])
    return 0


def _growth_history_prediction_weight(config: Any) -> float:
    try:
        return max(0.0, float(getattr(config, "growth_history_prediction_weight", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _growth_history_prediction_scale(config: Any) -> float:
    try:
        return max(1.0e-9, float(getattr(config, "growth_history_prediction_scale", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _growth_history_prediction_clip(config: Any) -> float:
    try:
        return max(0.0, float(getattr(config, "growth_history_prediction_clip", 8.0)))
    except (TypeError, ValueError):
        return 8.0


def _growth_history_prediction_min_edges(config: Any) -> int:
    try:
        return max(1, int(getattr(config, "growth_history_prediction_min_edges", 1)))
    except (TypeError, ValueError):
        return 1
