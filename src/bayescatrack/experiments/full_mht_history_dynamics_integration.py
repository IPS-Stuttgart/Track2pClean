"""Opt-in terminal motion-history objective for FullMHT.

FullMHT already scores scan assignments with label-free pairwise features.  This
module adds a second, terminal question: does the selected identity history look
internally coherent across its own edges?  The risk is label-free and compares
each edge against the other edges in the same seed-anchored history using the
same registration, growth, and local-deformation diagnostics used by the base
runner.

The objective is intentionally installed as an opt-in hook.  Frozen baseline rows
are unchanged unless a manifest or runner attaches ``terminal_motion_history_weight``
to ``FullMHTConfig`` and calls ``install_full_mht_history_dynamics_objective``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_REGISTERED_IOU_DROP = 0.15
_SHIFTED_IOU_DROP = 0.20
_GROWTH_RESIDUAL_OFFSET = 2.50
_GROWTH_MAHALANOBIS_OFFSET = 2.00
_LOCAL_DEFORMATION_OFFSET = 0.35
_MISSING_FEATURE_PENALTY = 1.00


@dataclass(frozen=True)
class HistoryEdgeFeatures:
    """Label-free diagnostics for one selected edge in an identity history."""

    registered_iou: float
    shifted_iou: float
    growth_residual: float
    growth_mahalanobis: float
    local_deformation: float


def install_full_mht_history_dynamics_objective() -> None:
    """Install terminal history-dynamics reranking into FullMHT."""

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    if getattr(full_mht, "_bayescatrack_history_dynamics_objective", False):
        return

    original_select = full_mht._select_final_hypothesis

    def _select_final_hypothesis_with_history_dynamics(
        hypotheses: Sequence[Any],
        *,
        sessions: Sequence[Any],
        feature_cache: Any,
        config: Any,
        track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    ) -> tuple[Any, dict[str, float | int]]:
        weight = _terminal_motion_history_weight(config)
        if weight <= 0.0:
            return original_select(
                hypotheses,
                sessions=sessions,
                feature_cache=feature_cache,
                config=config,
                track2p_prior_edges=track2p_prior_edges,
            )
        if not hypotheses:
            return original_select(
                hypotheses,
                sessions=sessions,
                feature_cache=feature_cache,
                config=config,
                track2p_prior_edges=track2p_prior_edges,
            )

        prior_weight = max(0.0, float(getattr(config, "terminal_history_risk_weight", 0.0)))
        best = hypotheses[0]
        best_rank = 1
        best_prior_risk = 0.0
        best_identity_risk = 0.0
        best_motion_risk = 0.0
        best_adjusted = -float("inf")

        for rank_index, hypothesis in enumerate(hypotheses, start=1):
            prior_risk = float(
                full_mht._terminal_history_risk(
                    hypothesis,
                    sessions=sessions,
                    feature_cache=feature_cache,
                    config=config,
                    track2p_prior_edges=track2p_prior_edges,
                )
            )
            identity_risk = float(
                full_mht._terminal_identity_history_risk(hypothesis, config=config)
            )
            motion_risk = terminal_motion_history_risk(
                hypothesis,
                sessions=sessions,
                feature_cache=feature_cache,
                config=config,
                track2p_prior_edges=track2p_prior_edges,
            )
            adjusted = (
                float(hypothesis.score)
                - prior_weight * prior_risk
                - identity_risk
                - weight * motion_risk
            )
            if adjusted > best_adjusted:
                best = hypothesis
                best_rank = int(rank_index)
                best_prior_risk = float(prior_risk)
                best_identity_risk = float(identity_risk)
                best_motion_risk = float(motion_risk)
                best_adjusted = float(adjusted)

        return best, {
            "terminal_history_risk": float(best_prior_risk),
            "terminal_identity_history_risk": float(best_identity_risk),
            "terminal_motion_history_risk": float(best_motion_risk),
            "terminal_motion_history_weight": float(weight),
            "terminal_adjusted_score": float(best_adjusted),
            "terminal_selected_rank": int(best_rank),
        }

    full_mht._select_final_hypothesis = (  # type: ignore[method-assign]
        _select_final_hypothesis_with_history_dynamics
    )
    full_mht._bayescatrack_history_dynamics_original_select = original_select
    full_mht._bayescatrack_history_dynamics_objective = True


def terminal_motion_history_risk(
    hypothesis: Any,
    *,
    sessions: Sequence[Any],
    feature_cache: Any,
    config: Any,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
) -> float:
    """Compute label-free outlier risk across complete identity histories."""

    tracks = np.asarray(getattr(hypothesis, "tracks", hypothesis), dtype=int)
    if tracks.ndim != 2 or tracks.size == 0:
        return 0.0

    risk = 0.0
    for row in tracks:
        observed_sessions = np.flatnonzero(np.asarray(row, dtype=int) >= 0)
        if observed_sessions.size < 3:
            continue
        features: list[HistoryEdgeFeatures] = []
        missing_features = 0
        for left, right in zip(observed_sessions[:-1], observed_sessions[1:]):
            edge_features = _edge_features_for_history_edge(
                sessions,
                feature_cache,
                source_session=int(left),
                target_session=int(right),
                source_roi=int(row[int(left)]),
                target_roi=int(row[int(right)]),
                config=config,
                track2p_prior_edges=track2p_prior_edges,
            )
            if edge_features is None:
                missing_features += 1
            else:
                features.append(edge_features)
        risk += row_motion_history_risk(
            features,
            missing_features=missing_features,
        )
    return float(risk)


def row_motion_history_risk(
    features: Sequence[HistoryEdgeFeatures], *, missing_features: int = 0
) -> float:
    """Return robust within-history outlier risk for edge diagnostics."""

    risk = max(0, int(missing_features)) * _MISSING_FEATURE_PENALTY
    if len(features) < 2:
        return float(risk)

    registered = _feature_array(features, "registered_iou")
    shifted = _feature_array(features, "shifted_iou")
    growth = _feature_array(features, "growth_residual")
    mahalanobis = _feature_array(features, "growth_mahalanobis")
    local = _feature_array(features, "local_deformation")

    risk += _low_outlier_risk(registered, allowed_drop=_REGISTERED_IOU_DROP)
    risk += _low_outlier_risk(shifted, allowed_drop=_SHIFTED_IOU_DROP)
    risk += _high_outlier_risk(growth, allowed_offset=_GROWTH_RESIDUAL_OFFSET)
    risk += _high_outlier_risk(mahalanobis, allowed_offset=_GROWTH_MAHALANOBIS_OFFSET)
    risk += _high_outlier_risk(local, allowed_offset=_LOCAL_DEFORMATION_OFFSET)
    return float(risk)


def _edge_features_for_history_edge(
    sessions: Sequence[Any],
    feature_cache: Any,
    *,
    source_session: int,
    target_session: int,
    source_roi: int,
    target_roi: int,
    config: Any,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
) -> HistoryEdgeFeatures | None:
    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    matrices = full_mht._sparse_pair_matrices(
        sessions,
        feature_cache,
        source_session=int(source_session),
        target_session=int(target_session),
        source_rois=(int(source_roi),),
        edge_top_k=max(1, int(getattr(config, "edge_top_k", 1))),
        config=config,
        track2p_prior_edges=track2p_prior_edges,
    )
    source_matches = np.flatnonzero(
        np.asarray(matrices.source_indices, dtype=int) == int(source_roi)
    )
    target_matches = np.flatnonzero(
        np.asarray(matrices.target_indices, dtype=int) == int(target_roi)
    )
    if source_matches.size == 0 or target_matches.size == 0:
        return None
    source_local = int(source_matches[0])
    target_local = int(target_matches[0])
    return HistoryEdgeFeatures(
        registered_iou=full_mht._finite_float(
            matrices.registered_iou[source_local, target_local], 0.0
        ),
        shifted_iou=full_mht._finite_float(
            matrices.shifted_iou[source_local, target_local], 0.0
        ),
        growth_residual=full_mht._finite_float(
            matrices.growth_residual[source_local, target_local], 0.0
        ),
        growth_mahalanobis=full_mht._finite_float(
            matrices.growth_mahalanobis[source_local, target_local], 0.0
        ),
        local_deformation=full_mht._finite_float(
            matrices.local_deformation[source_local, target_local], 0.0
        ),
    )


def _feature_array(features: Sequence[HistoryEdgeFeatures], name: str) -> np.ndarray:
    return np.asarray([float(getattr(feature, name)) for feature in features], dtype=float)


def _low_outlier_risk(values: np.ndarray, *, allowed_drop: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0
    reference = float(np.median(finite))
    return float(np.sum(np.maximum(0.0, reference - finite - float(allowed_drop))))


def _high_outlier_risk(values: np.ndarray, *, allowed_offset: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0
    reference = float(np.median(finite))
    return float(np.sum(np.maximum(0.0, finite - reference - float(allowed_offset))))


def _terminal_motion_history_weight(config: Any) -> float:
    try:
        return max(0.0, float(getattr(config, "terminal_motion_history_weight", 0.0)))
    except (TypeError, ValueError):
        return 0.0
