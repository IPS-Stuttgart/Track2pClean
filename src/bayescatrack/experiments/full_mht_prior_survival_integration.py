"""Opt-in calibrated prior-edge survival scoring for FullMHT.

This module wires the label-free prior-edge survival model into the existing
FullMHT scan-assignment score without changing the baseline runner.  Call
``install_full_mht_prior_survival_scoring`` before running FullMHT, then attach a
``track2p_prior_survival_weight`` attribute to the ``FullMHTConfig`` instance.
Positive survival log-ratios raise Track2p prior-edge survival likelihood;
negative log-ratios penalize suspicious proposal edges.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bayescatrack.experiments.full_mht_prior_survival_model import (
    PriorEdgeSurvivalConfig,
    PriorEdgeSurvivalDiagnostics,
    calibrate_prior_edge_survival_model,
)


def install_full_mht_prior_survival_scoring() -> None:
    """Install an opt-in calibrated survival term into FullMHT edge scoring."""

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    if getattr(full_mht, "_bayescatrack_prior_survival_scoring", False):
        return

    original_edge_score = full_mht._edge_score
    original_selected_edge_summary = full_mht._selected_edge_summary

    def _edge_score_with_prior_survival(
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
        survival_score = _prior_edge_survival_score(
            sessions,
            matrices,
            source_local=int(source_local),
            target_local=int(target_local),
            config=config,
            track2p_prior_edges=track2p_prior_edges,
            full_mht=full_mht,
        )
        if survival_score is None:
            return float(score)
        weight = float(getattr(config, "track2p_prior_survival_weight", 0.0))
        return float(score) + weight * float(survival_score)

    def _selected_edge_summary_with_prior_survival(
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
        if not int(output.get("is_track2p_prior", 0)):
            return output
        weight = float(getattr(config, "track2p_prior_survival_weight", 0.0))
        if weight == 0.0:
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
        survival_score = _prior_edge_survival_score(
            sessions,
            matrices,
            source_local=int(source_matches[0]),
            target_local=int(target_matches[0]),
            config=config,
            track2p_prior_edges=track2p_prior_edges,
            full_mht=full_mht,
        )
        if survival_score is None:
            output["track2p_prior_survival_score"] = "disabled"
            output["summary"] = f'{output["summary"]}|survival=disabled'
            return output
        weighted = weight * float(survival_score)
        output["track2p_prior_survival_score"] = float(survival_score)
        output["track2p_prior_survival_weighted_score"] = float(weighted)
        output["summary"] = (
            f'{output["summary"]}'
            f"|survival={full_mht._diagnostic_float(float(survival_score))}"
            f"|survival_weighted={full_mht._diagnostic_float(float(weighted))}"
        )
        return output

    full_mht._edge_score = _edge_score_with_prior_survival
    full_mht._selected_edge_summary = _selected_edge_summary_with_prior_survival
    full_mht._bayescatrack_prior_survival_original_edge_score = original_edge_score
    full_mht._bayescatrack_prior_survival_original_selected_edge_summary = (
        original_selected_edge_summary
    )
    full_mht._bayescatrack_prior_survival_scoring = True


def _prior_edge_survival_score(
    sessions: Sequence[Any],
    matrices: Any,
    *,
    source_local: int,
    target_local: int,
    config: Any,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    full_mht: Any,
) -> float | None:
    weight = float(getattr(config, "track2p_prior_survival_weight", 0.0))
    if weight == 0.0 or not track2p_prior_edges:
        return None
    source_roi = int(matrices.source_indices[int(source_local)])
    target_roi = int(matrices.target_indices[int(target_local)])
    edge = (
        int(matrices.source_session),
        int(matrices.target_session),
        source_roi,
        target_roi,
    )
    if edge not in track2p_prior_edges:
        return None
    diagnostics = _prior_edge_diagnostics_for_matrix(
        sessions,
        matrices,
        n_sessions=len(sessions),
        track2p_prior_edges=track2p_prior_edges,
        full_mht=full_mht,
    )
    model = calibrate_prior_edge_survival_model(
        diagnostics,
        config=_survival_config_from_full_mht_config(config),
    )
    if not model.enabled:
        return None
    edge_diag = _prior_edge_diagnostic(
        sessions,
        matrices,
        source_local=int(source_local),
        target_local=int(target_local),
        n_sessions=len(sessions),
        track2p_prior_edges=track2p_prior_edges,
        full_mht=full_mht,
    )
    return float(model.log_survival_ratio((edge_diag,))[0])


def _prior_edge_diagnostics_for_matrix(
    sessions: Sequence[Any],
    matrices: Any,
    *,
    n_sessions: int,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    full_mht: Any,
) -> tuple[PriorEdgeSurvivalDiagnostics, ...]:
    diagnostics: list[PriorEdgeSurvivalDiagnostics] = []
    source_indices = np.asarray(matrices.source_indices, dtype=int)
    target_indices = np.asarray(matrices.target_indices, dtype=int)
    for source_local, source_roi in enumerate(source_indices):
        for target_local, target_roi in enumerate(target_indices):
            edge = (
                int(matrices.source_session),
                int(matrices.target_session),
                int(source_roi),
                int(target_roi),
            )
            if edge not in track2p_prior_edges:
                continue
            diagnostics.append(
                _prior_edge_diagnostic(
                    sessions,
                    matrices,
                    source_local=int(source_local),
                    target_local=int(target_local),
                    n_sessions=int(n_sessions),
                    track2p_prior_edges=track2p_prior_edges,
                    full_mht=full_mht,
                )
            )
    return tuple(diagnostics)


def _prior_edge_diagnostic(
    sessions: Sequence[Any],
    matrices: Any,
    *,
    source_local: int,
    target_local: int,
    n_sessions: int,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    full_mht: Any,
) -> PriorEdgeSurvivalDiagnostics:
    source_roi = int(matrices.source_indices[int(source_local)])
    target_roi = int(matrices.target_indices[int(target_local)])
    edge = (
        int(matrices.source_session),
        int(matrices.target_session),
        source_roi,
        target_roi,
    )
    row_rank, column_rank = full_mht._edge_rank_values(
        matrices.registered_iou,
        source_local=int(source_local),
        target_local=int(target_local),
    )
    cell_a = full_mht._cell_probability(
        sessions, int(matrices.source_session), int(source_roi)
    )
    cell_b = full_mht._cell_probability(
        sessions, int(matrices.target_session), int(target_roi)
    )
    return PriorEdgeSurvivalDiagnostics(
        registered_iou=full_mht._finite_float(
            matrices.registered_iou[int(source_local), int(target_local)], 0.0
        ),
        shifted_iou=full_mht._finite_float(
            matrices.shifted_iou[int(source_local), int(target_local)], 0.0
        ),
        growth_residual=full_mht._finite_float(
            matrices.growth_residual[int(source_local), int(target_local)], 0.0
        ),
        growth_mahalanobis=full_mht._finite_float(
            matrices.growth_mahalanobis[int(source_local), int(target_local)], 0.0
        ),
        min_cell_probability=min(float(cell_a), float(cell_b)),
        area_ratio=full_mht._finite_float(
            matrices.area_ratio[int(source_local), int(target_local)], 1.0
        ),
        local_deformation=full_mht._finite_float(
            matrices.local_deformation[int(source_local), int(target_local)], 0.0
        ),
        row_rank=int(row_rank),
        column_rank=int(column_rank),
        terminal_edge=full_mht._prior_edge_is_terminal(
            edge, track2p_prior_edges=track2p_prior_edges
        ),
        last_session_edge=int(edge[1]) == int(n_sessions) - 1,
        complete_component=full_mht._prior_component_session_count(
            edge, track2p_prior_edges=track2p_prior_edges
        )
        >= int(n_sessions),
    )


def _survival_config_from_full_mht_config(config: Any) -> PriorEdgeSurvivalConfig:
    defaults = PriorEdgeSurvivalConfig()
    return PriorEdgeSurvivalConfig(
        min_anchor_registered_iou=float(
            getattr(
                config,
                "track2p_prior_survival_min_anchor_registered_iou",
                defaults.min_anchor_registered_iou,
            )
        ),
        min_anchor_shifted_iou=float(
            getattr(
                config,
                "track2p_prior_survival_min_anchor_shifted_iou",
                defaults.min_anchor_shifted_iou,
            )
        ),
        max_anchor_growth_mahalanobis=float(
            getattr(
                config,
                "track2p_prior_survival_max_anchor_growth_mahalanobis",
                defaults.max_anchor_growth_mahalanobis,
            )
        ),
        max_anchor_growth_residual=float(
            getattr(
                config,
                "track2p_prior_survival_max_anchor_growth_residual",
                defaults.max_anchor_growth_residual,
            )
        ),
        min_anchor_cell_probability=float(
            getattr(
                config,
                "track2p_prior_survival_min_anchor_cell_probability",
                defaults.min_anchor_cell_probability,
            )
        ),
        max_anchor_rank=int(
            getattr(
                config,
                "track2p_prior_survival_max_anchor_rank",
                defaults.max_anchor_rank,
            )
        ),
        max_background_registered_iou=float(
            getattr(
                config,
                "track2p_prior_survival_max_background_registered_iou",
                defaults.max_background_registered_iou,
            )
        ),
        max_background_shifted_iou=float(
            getattr(
                config,
                "track2p_prior_survival_max_background_shifted_iou",
                defaults.max_background_shifted_iou,
            )
        ),
        min_background_growth_mahalanobis=float(
            getattr(
                config,
                "track2p_prior_survival_min_background_growth_mahalanobis",
                defaults.min_background_growth_mahalanobis,
            )
        ),
        min_background_growth_residual=float(
            getattr(
                config,
                "track2p_prior_survival_min_background_growth_residual",
                defaults.min_background_growth_residual,
            )
        ),
        max_background_cell_probability=float(
            getattr(
                config,
                "track2p_prior_survival_max_background_cell_probability",
                defaults.max_background_cell_probability,
            )
        ),
        min_examples_per_class=int(
            getattr(
                config,
                "track2p_prior_survival_min_examples_per_class",
                defaults.min_examples_per_class,
            )
        ),
        min_feature_scale=float(
            getattr(
                config,
                "track2p_prior_survival_min_feature_scale",
                defaults.min_feature_scale,
            )
        ),
        per_feature_clip=float(
            getattr(
                config,
                "track2p_prior_survival_per_feature_clip",
                defaults.per_feature_clip,
            )
        ),
        score_clip=float(
            getattr(
                config,
                "track2p_prior_survival_score_clip",
                defaults.score_clip,
            )
        ),
    )
