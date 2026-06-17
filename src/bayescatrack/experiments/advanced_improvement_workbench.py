"""Command-line workbench for advanced BayesCaTrack improvement experiments.

This module is intentionally standalone.  It can be invoked with
``python -m bayescatrack.experiments.advanced_improvement_workbench`` without
modifying the top-level CLI while the advanced workstreams are reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ActiveLabelConfig:
    """Weights for active manual-label candidate ranking."""

    uncertainty_weight: float = 1.0
    disagreement_weight: float = 1.0
    margin_weight: float = 1.0
    missing_edge_weight: float = 1.0
    max_rows: int = 500


@dataclass(frozen=True)
class StratifiedMetricConfig:
    """Configuration for stratified benchmark summaries."""

    group_fields: tuple[str, ...]
    metric_fields: tuple[str, ...]


def select_active_label_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: ActiveLabelConfig | None = None,
) -> list[dict[str, Any]]:
    """Rank edges/tracks that are most informative for additional manual labels."""

    cfg = config or ActiveLabelConfig()
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = active_label_score(row, config=cfg)
        out = dict(row)
        out["active_label_score"] = score
        scored.append(out)
    scored.sort(key=lambda item: (-float(item["active_label_score"]), str(item)))
    return scored[: int(cfg.max_rows)]


def active_label_score(
    row: Mapping[str, Any], *, config: ActiveLabelConfig | None = None
) -> float:
    """Return an active-label priority score from edge-ranking/teacher rows."""

    cfg = config or ActiveLabelConfig()
    row_margin = _safe_float(row.get("row_margin"), np.nan)
    column_margin = _safe_float(row.get("column_margin"), np.nan)
    finite_margins = [
        value for value in (row_margin, column_margin) if np.isfinite(value)
    ]
    if finite_margins:
        low_margin_score = 1.0 / (1.0 + max(float(np.mean(finite_margins)), 0.0))
    else:
        low_margin_score = 1.0

    disagreement = 0.0
    if "in_ground_truth" in row and "in_track2p" in row:
        disagreement += float(
            _safe_bool(row.get("in_ground_truth")) != _safe_bool(row.get("in_track2p"))
        )
    if "in_ground_truth" in row and "in_bayes" in row:
        disagreement += float(
            _safe_bool(row.get("in_ground_truth")) != _safe_bool(row.get("in_bayes"))
        )
    if "in_track2p" in row and "in_bayes" in row:
        disagreement += 0.5 * float(
            _safe_bool(row.get("in_track2p")) != _safe_bool(row.get("in_bayes"))
        )

    missing_edge = float(str(row.get("missing_reason", "")).strip() != "")
    true_score = _safe_float(row.get("true_score"), np.nan)
    uncertainty = 1.0 if not np.isfinite(true_score) else 1.0 / (1.0 + abs(true_score))

    return float(
        cfg.margin_weight * low_margin_score
        + cfg.disagreement_weight * disagreement
        + cfg.missing_edge_weight * missing_edge
        + cfg.uncertainty_weight * uncertainty
    )


def stratified_metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: StratifiedMetricConfig,
) -> list[dict[str, Any]]:
    """Aggregate benchmark metrics by arbitrary metadata fields."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in config.group_fields)
        groups[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        summary: dict[str, Any] = {
            field: value for field, value in zip(config.group_fields, key, strict=True)
        }
        summary["rows"] = len(group_rows)
        for metric in config.metric_fields:
            values = np.asarray(
                [_safe_float(row.get(metric), np.nan) for row in group_rows],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else float("nan")
            )
            summary[f"{metric}_median"] = (
                float(np.median(values)) if values.size else float("nan")
            )
            summary[f"{metric}_min"] = (
                float(np.min(values)) if values.size else float("nan")
            )
            summary[f"{metric}_max"] = (
                float(np.max(values)) if values.size else float("nan")
            )
        summaries.append(summary)
    return summaries


def synthetic_stress_manifest(
    *,
    data_root: str,
    output_root: str,
    reference_root: str | None = None,
) -> dict[str, Any]:
    """Return a benchmark manifest covering controlled stress-test variants."""

    defaults: dict[str, Any] = {
        "data": data_root,
        "method": "global-assignment",
        "reference_kind": "manual-gt",
        "input_format": "suite2p",
        "include_non_cells": True,
        "weighted_masks": True,
        "transform_type": "fov-affine",
        "max_gap": 2,
        "format": "csv",
    }
    if reference_root is not None:
        defaults["reference"] = reference_root
    runs = []
    for transform in ("fov-affine", "local-affine-grid", "bspline"):
        for cost in ("registered-soft-iou", "roi-aware-shifted", "calibrated"):
            run: dict[str, Any] = {
                "name": f"stress-{transform}-{cost}",
                "transform_type": transform,
                "cost": cost,
                "output": f"{output_root}/stress-{transform}-{cost}.csv",
            }
            if cost == "calibrated":
                run["split"] = "leave-one-subject-out"
            runs.append(run)
    return {
        "defaults": defaults,
        "runs": runs,
        "comparisons": [
            {
                "name": "stress-summary",
                "inputs": {run["name"]: run["name"] for run in runs},
                "output": f"{output_root}/stress-summary.md",
                "highlight_best": True,
            }
        ],
    }


def track2p_result_improvement_manifest(
    *,
    data_root: str,
    output_root: str,
    reference_root: str | None = None,
    max_gap: int = 2,
    transform_type: str = "fov-affine",
    include_experimental_policy_dp: bool = False,
) -> dict[str, Any]:
    """Return a ready-to-run manifest for the highest-leverage result variants.

    The generated suite promotes the validated Track2p-policy minimum-threshold
    row as the current high-quality BayesCaTrack result and wires together the
    highest-leverage policy variants and remaining result-improvement directions
    that are already exposed elsewhere in the package: DP-rescued and prune-only
    Track2p-policy variants, solver-prior sweeps, residual-overlap costs,
    higher-order consistency, activity tie-breaking, local-evidence calibrated
    features, configurable hard negatives, monotone ranking costs, and registration QA.

    It intentionally emits a manifest instead of running the benchmarks directly
    so that long LOSO jobs can be reviewed, edited, or scheduled before launch.
    """

    from bayescatrack.experiments.track2p_loso_calibration import (
        calibration_feature_names,
    )

    max_gap = int(max_gap)
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")

    local_evidence_similarity_features = (
        "one_minus_weighted_dice",
        "one_minus_overlap_min_fraction",
    )
    local_evidence_features = tuple(
        dict.fromkeys(
            (
                *calibration_feature_names("default+local-evidence"),
                *local_evidence_similarity_features,
            )
        )
    )
    activity_local_evidence_features = tuple(
        dict.fromkeys(
            (
                *calibration_feature_names("default+activity+local-evidence"),
                *local_evidence_similarity_features,
            )
        )
    )

    defaults: dict[str, Any] = {
        "data": data_root,
        "reference_kind": "manual-gt",
        "input_format": "suite2p",
        "include_non_cells": True,
        "include_behavior": False,
        "weighted_masks": True,
        "weighted_centroids": True,
        "transform_type": transform_type,
        "auto_registration_candidates": (
            "none",
            "fov-translation",
            "fov-affine",
            "affine",
            "rigid",
        ),
        "fov_affine_mask_warp_mode": "bilinear",
        "seed_sessions": "all",
        "max_gap": max_gap,
        "format": "csv",
        "progress": True,
    }
    if reference_root is not None:
        defaults["reference"] = reference_root

    shifted_kwargs = {
        "shifted_iou_radius": 2,
        "shifted_iou_shift_penalty_weight": 0.25,
    }
    tuned_solver_priors = {
        "start_cost": 1.0,
        "end_cost": 1.0,
        "gap_penalty": 0.6,
        "cost_threshold": 2.0,
    }
    uncertainty_config = {
        "temperature": 2.0,
        "uncertainty_penalty_weight": 0.5,
        "registration_rmse_weight": 0.10,
        "invalid_warp_fraction_weight": 1.0,
        "empty_registered_roi_weight": 6.0,
        "gated_edge_weight": 6.0,
        "local_margin_weight": 0.5,
    }
    candidate_pruning_config = {
        "row_top_k": 20,
        "column_top_k": 20,
        "probability_threshold": 0.02,
    }
    dynamic_edge_prior_config = {
        "session_gap_weight": 0.25,
        "cell_probability_weight": 0.25,
        "area_ratio_weight": 0.10,
        "registration_empty_roi_weight": 8.0,
    }
    track2p_policy_prior_config = {
        "threshold_method": "min",
        "relief": 0.75,
        "accepted_cost_cap": 0.5,
        "non_policy_penalty": 0.05,
        "min_cost": -1.0,
        "consecutive_only": False,
        "row_top_k": 2,
        "rescue_min_iou": 0.10,
        "rescue_margin": 0.15,
    }
    track2p_policy_prune_config = {
        "prune_threshold_margin": 0.02,
        "prune_competition_margin": 0.02,
        "prune_min_area_ratio": 0.45,
        "prune_centroid_distance": 10.0,
    }
    track2p_policy_component_config = {
        "apply_splits": True,
        "threshold_margin_scale": 0.10,
        "competition_margin_scale": 0.20,
        "area_ratio_floor": 0.45,
        "centroid_distance_scale": 4.0,
        "split_risk_threshold": 1.50,
        "split_penalty": 0.25,
        "min_side_observations": 2,
    }
    track2p_policy_suffix_config = {
        "transform_type": "affine",
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "cell_probability_threshold": 0.5,
        "max_gap": 1,
        "weighted_masks": False,
        "weighted_centroids": False,
        "exclude_overlapping_pixels": False,
        **{
            key: value
            for key, value in track2p_policy_component_config.items()
            if key != "apply_splits"
        },
        "suffix_path_length": 2,
        "min_cell_probability": 0.80,
        "min_area_ratio": 0.80,
        "max_centroid_distance": 6.0,
        "min_shifted_iou": 0.30,
        "min_motion_consistency": 0.50,
        "min_shape_consistency": 0.82,
        "max_stitches_per_subject": 1,
    }
    track2p_policy_growth_veto_config = {
        **track2p_policy_suffix_config,
        "anchor_min_registered_iou": 0.50,
        "anchor_min_shifted_iou": 0.30,
        "anchor_min_cell_probability": 0.80,
        # Growth-veto is intentionally a tiny post-teacher row: after the
        # CoherenceSuffixTeacherRescue lead, the growth-field audit exposed one
        # extreme terminal false-continuation pocket. Keep the generated suite
        # aligned with the frozen strict gate rather than making it a broad
        # clean-up pass.
        "min_growth_residual_mahalanobis": 20.0,
        "min_growth_residual": 2.5,
        "min_veto_cell_probability": 0.50,
        "min_veto_registered_iou": 0.45,
        "max_veto_registered_iou": 0.60,
        "min_veto_shifted_iou": 0.60,
        "max_veto_shifted_iou": 0.80,
        "max_veto_min_cell_probability": 0.65,
        "max_vetoes_per_subject": 1,
    }
    teacher_prior_config = {
        "relief": 0.75,
        "teacher_cost_cap": 0.5,
        "non_teacher_penalty": 0.0,
        "min_cost": -1.0,
    }
    solver_prior_search = {
        "start_costs": (0.5, 1.0, 1.5, 2.0),
        "end_costs": (0.5, 1.0, 1.5, 2.0),
        "gap_penalties": (0.0, 0.3, 0.6, 0.9, 1.2),
        "cost_thresholds": (1.5, 2.0, 2.5, None),
        "objective": "complete_track_f1",
    }
    hard_negative_options = {
        "negative_to_positive_ratio": 8.0,
        "candidate_top_k_per_anchor": 30,
        "include_column_candidates": True,
        "hardness_feature_names": (
            "mahalanobis_centroid_distance",
            "centroid_distance",
            "one_minus_iou",
            "one_minus_mask_cosine",
            "activity_similarity_cost",
        ),
    }

    runs: list[dict[str, Any]] = [
        {
            "name": "track2p-baseline",
            "method": "track2p-baseline",
            "weighted_masks": False,
            "weighted_centroids": False,
            "output": f"{output_root}/track2p_baseline.csv",
        },
        {
            "name": "track2p-policy",
            "runner": "track2p-policy",
            "transform_type": "affine",
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "max_gap": 1,
            "weighted_masks": False,
            "weighted_centroids": False,
            "exclude_overlapping_pixels": False,
            "output": f"{output_root}/track2p_policy.csv",
        },
        {
            "name": "track2p-policy-component-cleanup",
            "runner": "track2p-policy-component-audit",
            "transform_type": "affine",
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "max_gap": 1,
            "weighted_masks": False,
            "weighted_centroids": False,
            "exclude_overlapping_pixels": False,
            **track2p_policy_component_config,
            "output": f"{output_root}/track2p_policy_component_cleanup.csv",
        },
        {
            "name": "track2p-policy-coherence-suffix-stitch",
            "runner": "track2p-policy-coherence-suffix-stitch",
            **track2p_policy_suffix_config,
            "output": f"{output_root}/track2p_policy_coherence_suffix_stitch.csv",
        },
        {
            "name": "track2p-policy-coherence-suffix-teacher-rescue",
            "runner": "track2p-policy-coherence-suffix-teacher-rescue",
            **track2p_policy_suffix_config,
            "teacher_edge_order": "structural",
            "teacher_action_filter": "all",
            "teacher_feature_preset": "none",
            "max_applied_teacher_edits": -1,
            "output": (
                f"{output_root}/" "track2p_policy_coherence_suffix_teacher_rescue.csv"
            ),
        },
        {
            "name": "track2p-policy-growth-veto-cleanup",
            "runner": "track2p-policy-growth-veto-cleanup",
            **track2p_policy_growth_veto_config,
            "output": f"{output_root}/track2p_policy_growth_veto_cleanup.csv",
        },
        {
            "name": "track2p-policy-coherence-suffix-growth-veto-cleanup",
            "runner": "track2p-policy-coherence-suffix-growth-veto-cleanup",
            **track2p_policy_growth_veto_config,
            "max_veto_local_neighbor_distortion": None,
            "output": (
                f"{output_root}/"
                "track2p_policy_coherence_suffix_growth_veto_cleanup.csv"
            ),
        },
        {
            "name": "track2p-policy-teacher-adjacent-rescue",
            "runner": "track2p-policy-teacher-adjacent-rescue",
            "transform_type": "affine",
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "max_gap": 1,
            "weighted_masks": False,
            "weighted_centroids": False,
            "exclude_overlapping_pixels": False,
            **{
                key: value
                for key, value in track2p_policy_component_config.items()
                if key != "apply_splits"
            },
            # Residual audits show that full gap/teacher propagation is too
            # permissive: the useful signal is concentrated in adjacent
            # Track2p-supported FNs and missing-seed backfills. Keep this canned
            # row in that narrow regime instead of running broad teacher rescue.
            "teacher_repair_preset": "residual-union-action-specific",
            "allow_completing_rescue": False,
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "allow_fragment_merges": False,
            "teacher_edge_order": "dynamic-seed-cell-confidence",
            "teacher_action_filter": "target-extension-or-seed-source-backfill",
            "teacher_feature_preset": "none",
            "target_extension_feature_preset": "moderate-iou-cell-confidence",
            "seed_source_feature_preset": "seed-source-cell-confident",
            "max_applied_edits": 3,
            "output": f"{output_root}/track2p_policy_teacher_adjacent_rescue.csv",
        },
        {
            "name": "track2p-policy-dp",
            "runner": "track2p-policy-dp",
            "transform_type": "affine",
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "row_top_k": 2,
            "rescue_min_iou": 0.10,
            "threshold_rescue_margin": 0.15,
            "accepted_bonus": 0.25,
            "rescue_penalty": 0.25,
            "gap_penalty": 1.0,
            "threshold_margin_weight": 0.5,
            "beam_width": 8,
            "max_gap": 2,
            "weighted_masks": False,
            "weighted_centroids": False,
            "exclude_overlapping_pixels": False,
            "output": f"{output_root}/track2p_policy_dp.csv",
        },
        {
            "name": "track2p-policy-pruned",
            "runner": "track2p-policy-pruned",
            "transform_type": "affine",
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "max_gap": 1,
            "weighted_masks": False,
            "weighted_centroids": False,
            "exclude_overlapping_pixels": False,
            **track2p_policy_prune_config,
            "output": f"{output_root}/track2p_policy_pruned.csv",
        },
        {
            "name": "oracle-gt-links",
            "method": "oracle-gt-links",
            "weighted_masks": False,
            "weighted_centroids": False,
            "output": f"{output_root}/oracle_gt_links.csv",
        },
        {
            "name": "global-registered-iou-prior-sweep",
            "method": "global-assignment",
            "cost": "registered-iou",
            "weighted_masks": False,
            "weighted_centroids": False,
            "sweep_start_costs": "0.5,1,2,5",
            "sweep_end_costs": "0.5,1,2,5",
            "sweep_gap_penalties": "0,0.6,1.2",
            "sweep_cost_thresholds": "1.5,2,4,6,none",
            "output": f"{output_root}/global_registered_iou_prior_sweep.csv",
        },
        {
            "name": "registered-shifted-iou-tuned",
            "method": "global-assignment",
            "cost": "registered-shifted-iou",
            "pairwise_cost_kwargs": shifted_kwargs,
            **tuned_solver_priors,
            "output": f"{output_root}/registered_shifted_iou_tuned.csv",
        },
        {
            "name": "roi-aware-shifted-tuned",
            "method": "global-assignment",
            "cost": "roi-aware-shifted",
            **tuned_solver_priors,
            "output": f"{output_root}/roi_aware_shifted_tuned.csv",
        },
        {
            "name": "roi-aware-shifted-auto-registration",
            "method": "global-assignment",
            "cost": "roi-aware-shifted",
            "transform_type": "auto",
            **tuned_solver_priors,
            "candidate_pruning_config": candidate_pruning_config,
            "output": f"{output_root}/roi_aware_shifted_auto_registration.csv",
        },
        {
            "name": "roi-aware-shifted-higher-order",
            "method": "global-assignment",
            "cost": "roi-aware-shifted",
            **tuned_solver_priors,
            "higher_order_triplet_weight": 0.5,
            "candidate_pruning_config": candidate_pruning_config,
            "higher_order_support_top_k": 8,
            "higher_order_support_cost_cap": 4.0,
            "higher_order_max_penalty": 2.0,
            "output": f"{output_root}/roi_aware_shifted_higher_order.csv",
        },
        {
            "name": "roi-aware-shifted-uncertainty-pruned",
            "method": "global-assignment",
            "cost": "roi-aware-shifted",
            **tuned_solver_priors,
            "edge_uncertainty_config": uncertainty_config,
            "candidate_pruning_config": candidate_pruning_config,
            "output": f"{output_root}/roi_aware_shifted_uncertainty_pruned.csv",
        },
        {
            "name": "roi-aware-shifted-dynamic-priors",
            "method": "global-assignment",
            "cost": "roi-aware-shifted",
            **tuned_solver_priors,
            "dynamic_edge_prior_config": dynamic_edge_prior_config,
            "edge_uncertainty_config": uncertainty_config,
            "candidate_pruning_config": candidate_pruning_config,
            "output": f"{output_root}/roi_aware_shifted_dynamic_priors.csv",
        },
        {
            "name": "roi-aware-shifted-track2p-policy-prior",
            "method": "global-assignment",
            "cost": "roi-aware-shifted",
            **tuned_solver_priors,
            "dynamic_edge_prior_config": dynamic_edge_prior_config,
            "edge_uncertainty_config": uncertainty_config,
            "candidate_pruning_config": candidate_pruning_config,
            "track2p_policy_prior_config": track2p_policy_prior_config,
            "output": f"{output_root}/roi_aware_shifted_track2p_policy_prior.csv",
        },
        {
            "name": "roi-aware-shifted-track2p-teacher-prior",
            "method": "global-assignment",
            "cost": "roi-aware-shifted",
            **tuned_solver_priors,
            "dynamic_edge_prior_config": dynamic_edge_prior_config,
            "edge_uncertainty_config": uncertainty_config,
            "candidate_pruning_config": candidate_pruning_config,
            "track2p_teacher_prior_config": teacher_prior_config,
            "output": f"{output_root}/roi_aware_shifted_track2p_teacher_prior.csv",
        },
        {
            "name": "roi-aware-shifted-learned-solver-priors",
            "runner": "track2p-solver-prior-loso",
            "cost": "roi-aware-shifted",
            "split": "leave-one-subject-out",
            "candidate_pruning_config": candidate_pruning_config,
            "dynamic_edge_prior_config": dynamic_edge_prior_config,
            "edge_uncertainty_config": uncertainty_config,
            **solver_prior_search,
            "output": f"{output_root}/roi_aware_shifted_learned_solver_priors.csv",
        },
        {
            "name": "roi-aware-shifted-activity-tiebreaker",
            "method": "global-assignment",
            "cost": "roi-aware-shifted",
            **tuned_solver_priors,
            "activity_tie_breaker_weight": 0.03,
            "activity_tie_breaker_component": "activity_tiebreaker_cost",
            "activity_trace_source": "auto",
            "output": f"{output_root}/roi_aware_shifted_activity_tiebreaker.csv",
        },
        {
            "name": "calibrated-loso-default",
            "method": "global-assignment",
            "cost": "calibrated",
            "split": "leave-one-subject-out",
            "calibration_feature_set": "default",
            "output": f"{output_root}/calibrated_loso_default.csv",
        },
        {
            "name": "calibrated-loso-local-evidence",
            "method": "global-assignment",
            "cost": "calibrated",
            "split": "leave-one-subject-out",
            "calibration_feature_set": "default+local-evidence",
            "output": f"{output_root}/calibrated_loso_local_evidence.csv",
        },
        {
            "name": "calibrated-loso-activity-local-evidence",
            "method": "global-assignment",
            "cost": "calibrated",
            "split": "leave-one-subject-out",
            "calibration_feature_set": "default+activity+local-evidence",
            "activity_trace_source": "auto",
            "output": f"{output_root}/calibrated_loso_activity_local_evidence.csv",
        },
        {
            "name": "configurable-loso-local-evidence-hgb",
            "runner": "track2p-loso-calibration",
            "feature_names": local_evidence_features,
            "calibration_model": "hist-gradient-boosting",
            "calibration_model_kwargs": {"random_state": 0, "max_iter": 200},
            "hard_negative_options": hard_negative_options,
            "output": f"{output_root}/configurable_loso_local_evidence_hgb.csv",
        },
        {
            "name": "configurable-loso-activity-local-evidence-logistic",
            "runner": "track2p-loso-calibration",
            "feature_names": activity_local_evidence_features,
            "sample_weight_strategy": "none",
            "calibration_model": "logistic",
            "calibration_model_kwargs": {"class_weight": None},
            "hard_negative_options": hard_negative_options,
            "output": f"{output_root}/configurable_loso_activity_local_evidence_logistic.csv",
        },
        {
            "name": "monotone-loso-local-evidence",
            "runner": "track2p-monotone-loso",
            "feature_names": local_evidence_features,
            "monotone_options": {
                "max_negatives_per_positive": 24,
                "max_iter": 1000,
                "binary_loss_weight": 0.05,
            },
            "output": f"{output_root}/monotone_loso_local_evidence.csv",
        },
        {
            "name": "registration-qa",
            "runner": "registration-qa",
            "cost": "roi-aware-shifted",
            "level": "summary",
            "output": f"{output_root}/registration_qa.csv",
        },
    ]

    if include_experimental_policy_dp:
        runs.insert(
            3,
            {
                "name": "track2p-policy-dp-experimental",
                "runner": "track2p-policy-dp",
                "transform_type": "affine",
                "threshold_method": "min",
                "iou_distance_threshold": 12.0,
                "cell_probability_threshold": 0.5,
                "row_top_k": 3,
                "rescue_min_iou": 0.05,
                "threshold_rescue_margin": 0.25,
                "accepted_bonus": 0.25,
                "rescue_penalty": 0.25,
                "gap_penalty": 1.0,
                "threshold_margin_weight": 0.5,
                "beam_width": 16,
                "max_gap": 2,
                "path_candidates_per_seed": 8,
                "path_selection_beam_width": 512,
                "weighted_masks": False,
                "weighted_centroids": False,
                "exclude_overlapping_pixels": False,
                "output": f"{output_root}/track2p_policy_dp_experimental.csv",
            },
        )

    comparable_runs = [
        run for run in runs if run.get("runner", "track2p") != "registration-qa"
    ]
    comparison_inputs = {run["name"]: run["name"] for run in comparable_runs}
    return {
        "defaults": defaults,
        "runs": runs,
        "comparisons": [
            {
                "name": "result-improvement-comparison",
                "inputs": comparison_inputs,
                "output": f"{output_root}/result_improvement_comparison.md",
                "highlight_best": True,
            },
            {
                "name": "result-improvement-comparison-csv",
                "inputs": comparison_inputs,
                "output": f"{output_root}/result_improvement_comparison.csv",
                "format": "csv",
            },
        ],
    }


def precision_recall_threshold_table(
    probabilities: Any,
    labels: Any,
    *,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float | int]]:
    """Return precision/recall/F1 over probability rejection thresholds."""

    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if probs.shape != y.shape:
        raise ValueError("probabilities and labels must have the same length")
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        pred = probs >= float(threshold)
        positive = y != 0
        tp = int(np.count_nonzero(pred & positive))
        fp = int(np.count_nonzero(pred & ~positive))
        fn = int(np.count_nonzero(~pred & positive))
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        rows.append(
            {
                "threshold": float(threshold),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": _ratio(2.0 * precision * recall, precision + recall),
            }
        )
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.advanced_improvement_workbench",
        description="Advanced diagnostics and manifest helpers for BayesCaTrack result improvement.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    active = subparsers.add_parser(
        "active-labels", help="Rank edge rows for additional manual labeling"
    )
    active.add_argument("--input", required=True, type=Path)
    active.add_argument("--output", required=True, type=Path)
    active.add_argument("--max-rows", type=int, default=500)

    stratify = subparsers.add_parser(
        "stratify", help="Aggregate benchmark metrics by metadata fields"
    )
    stratify.add_argument("--input", required=True, type=Path)
    stratify.add_argument("--output", required=True, type=Path)
    stratify.add_argument("--group-field", action="append", required=True)
    stratify.add_argument("--metric", action="append", required=True)

    stress = subparsers.add_parser(
        "stress-manifest", help="Write a synthetic stress-test benchmark manifest"
    )
    stress.add_argument("--data-root", required=True)
    stress.add_argument("--output-root", required=True)
    stress.add_argument("--reference-root", default=None)
    stress.add_argument("--output", required=True, type=Path)

    improvement = subparsers.add_parser(
        "track2p-improvement-manifest",
        help="Write a Track2p result-improvement benchmark manifest",
    )
    improvement.add_argument("--data-root", required=True)
    improvement.add_argument("--output-root", required=True)
    improvement.add_argument("--reference-root", default=None)
    improvement.add_argument("--max-gap", type=int, default=2)
    improvement.add_argument("--transform-type", default="fov-affine")
    improvement.add_argument(
        "--include-experimental-policy-dp",
        action="store_true",
        help="Also include the currently experimental Track2p-policy DP rescue row",
    )
    improvement.add_argument("--output", required=True, type=Path)

    pr = subparsers.add_parser(
        "pr-table", help="Build a precision/recall table from probability-label CSV"
    )
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    pr.add_argument("--probability-column", default="probability")
    pr.add_argument("--label-column", default="label")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "active-labels":
        rows = read_csv_rows(args.input)
        selected = select_active_label_candidates(
            rows, config=ActiveLabelConfig(max_rows=args.max_rows)
        )
        write_csv_rows(selected, args.output)
        return 0
    if args.command == "stratify":
        rows = read_csv_rows(args.input)
        summary = stratified_metric_summary(
            rows,
            config=StratifiedMetricConfig(
                group_fields=tuple(args.group_field),
                metric_fields=tuple(args.metric),
            ),
        )
        write_csv_rows(summary, args.output)
        return 0
    if args.command == "stress-manifest":
        manifest = synthetic_stress_manifest(
            data_root=args.data_root,
            output_root=args.output_root,
            reference_root=args.reference_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "track2p-improvement-manifest":
        manifest = track2p_result_improvement_manifest(
            data_root=args.data_root,
            output_root=args.output_root,
            reference_root=args.reference_root,
            max_gap=args.max_gap,
            transform_type=args.transform_type,
            include_experimental_policy_dp=args.include_experimental_policy_dp,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "pr-table":
        rows = read_csv_rows(args.input)
        probabilities = [
            _safe_float(row.get(args.probability_column), np.nan) for row in rows
        ]
        labels = [int(_safe_float(row.get(args.label_column), 0.0)) for row in rows]
        write_csv_rows(
            precision_recall_threshold_table(probabilities, labels), args.output
        )
        return 0
    raise ValueError(f"Unsupported command {args.command!r}")


def _safe_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if np.isfinite(numeric) else float(default)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"", "0", "false", "f", "no", "n", "none", "null", "nan"}:
            return False
        if text in {"1", "true", "t", "yes", "y"}:
            return True
    return bool(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
