"""Opt-in no-prior continuation likelihood scoring for FullMHT.

This module wires the label-free no-prior continuation model into FullMHT edge
scoring.  It only affects candidate edges that are not Track2p proposal edges and
whose source ROI has no Track2p proposal successor for the current scan.  Positive
log-ratios support opening a continuation despite the missing proposal successor;
negative log-ratios make the MHT beam prefer death/missed-detection hypotheses.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bayescatrack.experiments.full_mht_no_prior_continuation_model import (
    NoPriorContinuationConfig,
    NoPriorContinuationDiagnostics,
    calibrate_no_prior_continuation_model,
)


def install_full_mht_no_prior_continuation_scoring() -> None:
    """Install an opt-in no-prior continuation term into FullMHT scoring."""

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    if getattr(full_mht, "_bayescatrack_no_prior_continuation_scoring", False):
        return

    original_edge_score = full_mht._edge_score
    original_selected_edge_summary = full_mht._selected_edge_summary

    def _edge_score_with_no_prior_continuation(
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
        continuation_score = _no_prior_continuation_score(
            sessions,
            matrices,
            source_local=int(source_local),
            target_local=int(target_local),
            config=config,
            track2p_prior_edges=track2p_prior_edges,
            full_mht=full_mht,
        )
        if continuation_score is None:
            return float(score)
        weight = float(getattr(config, "no_prior_continuation_likelihood_weight", 0.0))
        return float(score) + weight * float(continuation_score)

    def _selected_edge_summary_with_no_prior_continuation(
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
        source_matches = np.flatnonzero(
            np.asarray(matrices.source_indices, dtype=int)
            == int(active_source.source_roi)
        )
        target_matches = np.flatnonzero(
            np.asarray(matrices.target_indices, dtype=int) == int(target_roi)
        )
        if source_matches.size == 0 or target_matches.size == 0:
            return output
        continuation_score = _no_prior_continuation_score(
            sessions,
            matrices,
            source_local=int(source_matches[0]),
            target_local=int(target_matches[0]),
            config=config,
            track2p_prior_edges=track2p_prior_edges,
            full_mht=full_mht,
        )
        if continuation_score is None:
            return output
        weight = float(getattr(config, "no_prior_continuation_likelihood_weight", 0.0))
        weighted = weight * float(continuation_score)
        output["no_prior_continuation_score"] = float(continuation_score)
        output["no_prior_continuation_weighted_score"] = float(weighted)
        output["summary"] = (
            f'{output["summary"]}'
            f"|no_prior_cont={full_mht._diagnostic_float(float(continuation_score))}"
            f"|no_prior_cont_weighted={full_mht._diagnostic_float(float(weighted))}"
        )
        return output

    full_mht._edge_score = _edge_score_with_no_prior_continuation
    full_mht._selected_edge_summary = _selected_edge_summary_with_no_prior_continuation
    full_mht._bayescatrack_no_prior_continuation_original_edge_score = original_edge_score
    full_mht._bayescatrack_no_prior_continuation_original_selected_edge_summary = (
        original_selected_edge_summary
    )
    full_mht._bayescatrack_no_prior_continuation_scoring = True


def _no_prior_continuation_score(
    sessions: Sequence[Any],
    matrices: Any,
    *,
    source_local: int,
    target_local: int,
    config: Any,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    full_mht: Any,
) -> float | None:
    weight = float(getattr(config, "no_prior_continuation_likelihood_weight", 0.0))
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
    if edge in track2p_prior_edges:
        return None
    if full_mht._has_prior_successor_for_roi(
        source_session=int(matrices.source_session),
        target_session=int(matrices.target_session),
        source_roi=source_roi,
        track2p_prior_edges=track2p_prior_edges,
    ):
        return None

    diagnostics = _no_prior_continuation_diagnostics_for_matrix(
        sessions,
        matrices,
        track2p_prior_edges=track2p_prior_edges,
        full_mht=full_mht,
    )
    model = calibrate_no_prior_continuation_model(
        diagnostics,
        config=_continuation_config_from_full_mht_config(config),
    )
    if not model.enabled:
        return None
    edge_diag = _no_prior_continuation_diagnostic(
        sessions,
        matrices,
        source_local=int(source_local),
        target_local=int(target_local),
        full_mht=full_mht,
    )
    return float(model.log_continuation_ratio((edge_diag,))[0])


def _no_prior_continuation_diagnostics_for_matrix(
    sessions: Sequence[Any],
    matrices: Any,
    *,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    full_mht: Any,
) -> tuple[NoPriorContinuationDiagnostics, ...]:
    diagnostics: list[NoPriorContinuationDiagnostics] = []
    source_indices = np.asarray(matrices.source_indices, dtype=int)
    target_indices = np.asarray(matrices.target_indices, dtype=int)
    for source_local, source_roi in enumerate(source_indices):
        if full_mht._has_prior_successor_for_roi(
            source_session=int(matrices.source_session),
            target_session=int(matrices.target_session),
            source_roi=int(source_roi),
            track2p_prior_edges=track2p_prior_edges,
        ):
            continue
        for target_local, target_roi in enumerate(target_indices):
            edge = (
                int(matrices.source_session),
                int(matrices.target_session),
                int(source_roi),
                int(target_roi),
            )
            if edge in track2p_prior_edges:
                continue
            diagnostics.append(
                _no_prior_continuation_diagnostic(
                    sessions,
                    matrices,
                    source_local=int(source_local),
                    target_local=int(target_local),
                    full_mht=full_mht,
                )
            )
    return tuple(diagnostics)


def _no_prior_continuation_diagnostic(
    sessions: Sequence[Any],
    matrices: Any,
    *,
    source_local: int,
    target_local: int,
    full_mht: Any,
) -> NoPriorContinuationDiagnostics:
    source_roi = int(matrices.source_indices[int(source_local)])
    target_roi = int(matrices.target_indices[int(target_local)])
    row_rank, column_rank = full_mht._edge_rank_values(
        matrices.registered_iou,
        source_local=int(source_local),
        target_local=int(target_local),
    )
    cell_a = full_mht._cell_probability(
        sessions,
        int(matrices.source_session),
        source_roi,
    )
    cell_b = full_mht._cell_probability(
        sessions,
        int(matrices.target_session),
        target_roi,
    )
    registered = full_mht._finite_float(
        matrices.registered_iou[int(source_local), int(target_local)],
        0.0,
    )
    return NoPriorContinuationDiagnostics(
        registered_iou=registered,
        shifted_iou=full_mht._finite_float(
            matrices.shifted_iou[int(source_local), int(target_local)],
            0.0,
        ),
        growth_residual=full_mht._finite_float(
            matrices.growth_residual[int(source_local), int(target_local)],
            0.0,
        ),
        growth_mahalanobis=full_mht._finite_float(
            matrices.growth_mahalanobis[int(source_local), int(target_local)],
            0.0,
        ),
        min_cell_probability=min(float(cell_a), float(cell_b)),
        area_ratio=full_mht._finite_float(
            matrices.area_ratio[int(source_local), int(target_local)],
            1.0,
        ),
        centroid_distance=full_mht._finite_float(
            matrices.centroid_distance[int(source_local), int(target_local)],
            0.0,
        ),
        threshold_margin=registered - float(matrices.threshold),
        local_deformation=full_mht._finite_float(
            matrices.local_deformation[int(source_local), int(target_local)],
            0.0,
        ),
        row_rank=int(row_rank),
        column_rank=int(column_rank),
    )


def _continuation_config_from_full_mht_config(config: Any) -> NoPriorContinuationConfig:
    defaults = NoPriorContinuationConfig()
    return NoPriorContinuationConfig(
        min_anchor_registered_iou=float(
            getattr(
                config,
                "no_prior_continuation_min_anchor_registered_iou",
                defaults.min_anchor_registered_iou,
            )
        ),
        min_anchor_shifted_iou=float(
            getattr(
                config,
                "no_prior_continuation_min_anchor_shifted_iou",
                defaults.min_anchor_shifted_iou,
            )
        ),
        max_anchor_growth_mahalanobis=float(
            getattr(
                config,
                "no_prior_continuation_max_anchor_growth_mahalanobis",
                defaults.max_anchor_growth_mahalanobis,
            )
        ),
        max_anchor_growth_residual=float(
            getattr(
                config,
                "no_prior_continuation_max_anchor_growth_residual",
                defaults.max_anchor_growth_residual,
            )
        ),
        min_anchor_cell_probability=float(
            getattr(
                config,
                "no_prior_continuation_min_anchor_cell_probability",
                defaults.min_anchor_cell_probability,
            )
        ),
        max_anchor_local_deformation=float(
            getattr(
                config,
                "no_prior_continuation_max_anchor_local_deformation",
                defaults.max_anchor_local_deformation,
            )
        ),
        max_anchor_rank=int(
            getattr(
                config,
                "no_prior_continuation_max_anchor_rank",
                defaults.max_anchor_rank,
            )
        ),
        max_background_registered_iou=float(
            getattr(
                config,
                "no_prior_continuation_max_background_registered_iou",
                defaults.max_background_registered_iou,
            )
        ),
        max_background_shifted_iou=float(
            getattr(
                config,
                "no_prior_continuation_max_background_shifted_iou",
                defaults.max_background_shifted_iou,
            )
        ),
        min_background_growth_mahalanobis=float(
            getattr(
                config,
                "no_prior_continuation_min_background_growth_mahalanobis",
                defaults.min_background_growth_mahalanobis,
            )
        ),
        min_background_growth_residual=float(
            getattr(
                config,
                "no_prior_continuation_min_background_growth_residual",
                defaults.min_background_growth_residual,
            )
        ),
        max_background_cell_probability=float(
            getattr(
                config,
                "no_prior_continuation_max_background_cell_probability",
                defaults.max_background_cell_probability,
            )
        ),
        min_background_local_deformation=float(
            getattr(
                config,
                "no_prior_continuation_min_background_local_deformation",
                defaults.min_background_local_deformation,
            )
        ),
        min_examples_per_class=int(
            getattr(
                config,
                "no_prior_continuation_min_examples_per_class",
                defaults.min_examples_per_class,
            )
        ),
        min_feature_scale=float(
            getattr(
                config,
                "no_prior_continuation_min_feature_scale",
                defaults.min_feature_scale,
            )
        ),
        per_feature_clip=float(
            getattr(
                config,
                "no_prior_continuation_per_feature_clip",
                defaults.per_feature_clip,
            )
        ),
        score_clip=float(
            getattr(
                config,
                "no_prior_continuation_score_clip",
                defaults.score_clip,
            )
        ),
    )
