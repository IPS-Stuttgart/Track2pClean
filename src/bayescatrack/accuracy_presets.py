"""Accuracy-oriented Track2p benchmark presets.

The presets in this module are intentionally conservative: they only combine
association knobs that are already implemented by the benchmark runner, and they
keep manual ground-truth evaluation as the default target.  The goal is to make
strong BayesCaTrack configurations reproducible instead of relying on ad-hoc
command-line JSON assembly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from bayescatrack.experiments.track2p_benchmark import (
    ReferenceKind,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    run_track2p_benchmark,
)

AccuracyPresetName = Literal[
    "registered-shifted-iou-safe",
    "roi-aware-shifted-pruned",
    "roi-aware-shifted-consensus",
    "track2p-stability-cleanup",
    "track2p-supported-gap-cleanup",
    "track2p-confidence-ordered-strict-gap-cleanup",
    "track2p-teacher-adjacent-rescue",
    "track2p-teacher-fn-rescue",
    "track2p-residual-union-action-specific-rescue",
    "track2p-completing-rescue-action-specific",
]
AccuracyPresetRunner = Literal[
    "benchmark",
    "stability-cleanup",
    "supported-gap-cleanup",
    "confidence-ordered-strict-gap-cleanup",
    "teacher-adjacent-rescue",
]


@dataclass(frozen=True)
class AccuracyPreset:
    """Named benchmark configuration intended for accuracy tuning."""

    name: AccuracyPresetName
    description: str
    config: Track2pBenchmarkConfig
    runner: AccuracyPresetRunner = "benchmark"
    runner_kwargs: Mapping[str, object] | None = None


def build_track2p_accuracy_presets(
    data: str | Path,
    *,
    reference: str | Path | None = None,
    reference_kind: ReferenceKind = "manual-gt",
    plane_name: str = "plane0",
    input_format: str = "auto",
    transform_type: str = "auto",
    auto_registration_candidates: Sequence[str] = (
        "none",
        "fov-translation",
        "fov-affine",
        "affine",
        "rigid",
    ),
    max_gap: int = 2,
    cost_threshold: float | None = 6.0,
    progress: bool = True,
) -> tuple[AccuracyPreset, ...]:
    """Return reproducible high-recall Track2p accuracy presets.

    These presets are meant as the next benchmark candidates after the plain
    registered-IoU/global-assignment baseline.  They emphasize recall-preserving
    candidate generation followed by conservative rank-aware structural
    penalties, which is usually preferable when optimizing complete-track F1.
    """

    base = Track2pBenchmarkConfig(
        data=Path(data),
        method="global-assignment",
        plane_name=plane_name,
        input_format=input_format,
        reference=None if reference is None else Path(reference),
        reference_kind=reference_kind,
        max_gap=int(max_gap),
        transform_type=transform_type,
        auto_registration_candidates=tuple(auto_registration_candidates),
        fov_affine_mask_warp_mode="bilinear",
        start_cost=5.0,
        end_cost=5.0,
        gap_penalty=1.0,
        cost_threshold=cost_threshold,
        include_behavior=True,
        include_non_cells=True,
        cell_probability_threshold=0.0,
        weighted_masks=True,
        exclude_overlapping_pixels=True,
        restrict_to_reference_seed_rois=True,
        progress=progress,
    )

    shifted_safe = replace(
        base,
        cost="registered-shifted-iou",
        pairwise_cost_kwargs={
            "shifted_iou_radius": 3,
            "shifted_iou_shift_penalty_weight": 0.15,
            "shifted_iou_shift_penalty_scale": 1.5,
        },
        higher_order_consistency_config={
            "triplet_weight": 0.20,
            "support_top_k": 8,
            "support_cost_cap": 4.0,
            "max_penalty": 1.0,
        },
    )

    pruned_roi_aware = replace(
        base,
        cost="roi-aware-shifted",
        pairwise_cost_kwargs={
            "local_evidence_components": True,
            "shifted_iou_radius": 3,
            "shifted_iou_shift_penalty_weight": 0.20,
            "shifted_iou_shift_penalty_scale": 1.5,
        },
        candidate_pruning_config={
            "row_top_k": 24,
            "column_top_k": 24,
            "max_cost": 6.0 if cost_threshold is None else float(cost_threshold),
        },
        track2p_policy_prior_config={
            "threshold_method": "min",
            "relief": 0.20,
            "accepted_cost_cap": 4.0,
            "non_policy_penalty": 0.05,
            "mutual_top_k": 2,
            "rescue_min_iou": 0.10,
            "rescue_margin": 0.05,
            "max_gap": max_gap,
        },
        dynamic_edge_prior_config={
            "session_gap_weight": 0.25,
            "cell_probability_weight": 0.50,
            "registration_empty_roi_weight": 8.0,
            "reciprocal_rank_weight": 0.15,
            "reciprocal_rank_cap": 0.75,
        },
        higher_order_consistency_config={
            "triplet_weight": 0.30,
            "support_top_k": 10,
            "support_cost_cap": 4.0,
            "max_penalty": 1.5,
        },
        activity_tie_breaker_weight=0.05,
    )

    consensus = replace(
        pruned_roi_aware,
        consensus_prior_config={
            "variant_costs": (
                "registered-iou",
                "registered-shifted-iou",
                "roi-aware-shifted",
            ),
            "min_votes": 2,
            "relief": 0.20,
            "max_relief": 0.80,
            "ignore_variant_failures": True,
        },
    )

    stability_cleanup = replace(
        base,
        transform_type="affine",
        max_gap=1,
        include_behavior=False,
        include_non_cells=False,
        cell_probability_threshold=0.5,
        weighted_masks=False,
        weighted_centroids=False,
        exclude_overlapping_pixels=False,
    )

    supported_gap_max_gap = max(2, int(max_gap))
    supported_gap_cleanup = replace(
        base,
        transform_type="affine",
        max_gap=supported_gap_max_gap,
        include_behavior=False,
        include_non_cells=False,
        cell_probability_threshold=0.5,
        weighted_masks=False,
        weighted_centroids=False,
        exclude_overlapping_pixels=False,
    )
    confidence_gap_runner_kwargs = {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "transform_type": "affine",
        "cell_probability_threshold": 0.5,
        "max_gap": supported_gap_max_gap,
        "cleanup_config_kwargs": {
            "split_risk_threshold": 1.50,
            "split_penalty": 0.25,
            "min_side_observations": 2,
            "require_complete_track": True,
        },
    }
    teacher_rescue_runner_kwargs = {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "transform_type": "affine",
        "cell_probability_threshold": 0.5,
        "allow_completing_rescue": False,
        "cleanup_config_kwargs": {
            "split_risk_threshold": 1.50,
            "split_penalty": 0.25,
            "min_side_observations": 2,
            "require_complete_track": True,
        },
    }
    teacher_fn_rescue_runner_kwargs = {
        **teacher_rescue_runner_kwargs,
        "teacher_repair_preset": "track2p-fn-high-confidence",
    }
    residual_union_action_specific_runner_kwargs = {
        **teacher_rescue_runner_kwargs,
        "teacher_repair_preset": "residual-union-action-specific",
    }
    completing_rescue_action_specific_runner_kwargs = {
        **teacher_rescue_runner_kwargs,
        "teacher_repair_preset": "completing-rescue-action-specific",
    }

    return (
        AccuracyPreset(
            name="registered-shifted-iou-safe",
            description=(
                "Shift-tolerant registered IoU with a small shift penalty and "
                "triplet support for skip edges."
            ),
            config=shifted_safe,
        ),
        AccuracyPreset(
            name="roi-aware-shifted-pruned",
            description=(
                "ROI-aware shifted-overlap costs with high-recall row/column "
                "candidate pruning, mutual-top-k Track2p-policy relief, dynamic "
                "rank-aware edge priors, activity tie-breaking, and higher-order "
                "consistency."
            ),
            config=pruned_roi_aware,
        ),
        AccuracyPreset(
            name="roi-aware-shifted-consensus",
            description=(
                "The pruned ROI-aware preset plus ensemble consensus relief for "
                "links independently recovered by multiple cost families."
            ),
            config=consensus,
        ),
        AccuracyPreset(
            name="track2p-stability-cleanup",
            description=(
                "Track2p-policy affine/min-threshold row with prune-only "
                "threshold-stability splitting for unstable adjacent bridges."
            ),
            config=stability_cleanup,
            runner="stability-cleanup",
            runner_kwargs={
                "threshold_method": "min",
                "transform_type": "affine",
                "cell_probability_threshold": 0.5,
                "cleanup_config_kwargs": {
                    "iou_distance_thresholds": (10.0, 12.0, 14.0),
                    "base_iou_distance_threshold": 12.0,
                    "min_support_fraction": 2.0 / 3.0,
                    "min_side_observations": 2,
                },
            },
        ),
        AccuracyPreset(
            name="track2p-supported-gap-cleanup",
            description=(
                "Track2p-style affine/min-threshold policy with support-gated "
                "gap rescue and weakest-bridge component cleanup."
            ),
            config=supported_gap_cleanup,
            runner="supported-gap-cleanup",
            runner_kwargs={
                **confidence_gap_runner_kwargs,
                "min_bridge_support": 1,
                "reject_conflicting_bridge_support": True,
                "apply_splits": True,
            },
        ),
        AccuracyPreset(
            name="track2p-confidence-ordered-strict-gap-cleanup",
            description=(
                "Component cleanup plus hard-gated gap rescue, applying accepted "
                "gap candidates by descending gate slack so weak candidates do "
                "not block stronger suffix merges."
            ),
            config=supported_gap_cleanup,
            runner="confidence-ordered-strict-gap-cleanup",
            runner_kwargs=confidence_gap_runner_kwargs,
        ),
        AccuracyPreset(
            name="track2p-teacher-adjacent-rescue",
            description=(
                "Component cleanup plus guarded adjacent Track2p-teacher rescue "
                "for residual false-negative links, blocking rescue edges that "
                "would complete otherwise incomplete rows."
            ),
            config=stability_cleanup,
            runner="teacher-adjacent-rescue",
            runner_kwargs=teacher_rescue_runner_kwargs,
        ),
        AccuracyPreset(
            name="track2p-teacher-fn-rescue",
            description=(
                "Component cleanup plus the dedicated Track2p-supported false-"
                "negative repair preset: dynamic-confidence target extensions, "
                "a high-confidence local feature gate, and a small edit budget."
            ),
            config=stability_cleanup,
            runner="teacher-adjacent-rescue",
            runner_kwargs=teacher_fn_rescue_runner_kwargs,
        ),
        AccuracyPreset(
            name="track2p-residual-union-action-specific-rescue",
            description=(
                "Component cleanup plus the residual-union teacher-rescue preset "
                "with action-specific gates: moderate-IoU/cell-confidence target "
                "extensions for Track2p-supported adjacent FNs, and stricter "
                "seed-source cell-confidence backfills for the missing seed-session "
                "ROI bucket."
            ),
            config=stability_cleanup,
            runner="teacher-adjacent-rescue",
            runner_kwargs=residual_union_action_specific_runner_kwargs,
        ),
        AccuracyPreset(
            name="track2p-completing-rescue-action-specific",
            description=(
                "Component cleanup plus a completing-only Track2p-teacher rescue "
                "preset: spend a tiny edit budget only on teacher edges that would "
                "complete a seed-anchored row, with action-specific target-extension "
                "and missing-seed source-backfill gates. This tests the residual "
                "complete-FN bucket without admitting metric-neutral teacher edits."
            ),
            config=stability_cleanup,
            runner="teacher-adjacent-rescue",
            runner_kwargs=completing_rescue_action_specific_runner_kwargs,
        ),
    )


def run_track2p_accuracy_presets(
    data: str | Path,
    *,
    reference: str | Path | None = None,
    reference_kind: ReferenceKind = "manual-gt",
    preset_names: Iterable[AccuracyPresetName] | None = None,
    **preset_kwargs: object,
) -> dict[str, list[SubjectBenchmarkResult]]:
    """Run selected accuracy presets and return results keyed by preset name."""

    presets = build_track2p_accuracy_presets(
        data,
        reference=reference,
        reference_kind=reference_kind,
        **preset_kwargs,
    )
    requested = None if preset_names is None else {str(name) for name in preset_names}
    results: dict[str, list[SubjectBenchmarkResult]] = {}
    for preset in presets:
        if requested is not None and preset.name not in requested:
            continue
        results[preset.name] = _run_accuracy_preset(preset)
    if requested is not None:
        missing = requested.difference(results)
        if missing:
            raise ValueError(
                f"Unknown accuracy preset(s): {', '.join(sorted(missing))}"
            )
    return results


def _run_accuracy_preset(preset: AccuracyPreset) -> list[SubjectBenchmarkResult]:
    if preset.runner == "benchmark":
        return run_track2p_benchmark(preset.config)
    if preset.runner == "stability-cleanup":
        from bayescatrack.experiments import (
            track2p_policy_stability_cleanup as stability_cleanup,
        )

        runner_kwargs = dict(preset.runner_kwargs or {})
        cleanup_kwargs = runner_kwargs.pop("cleanup_config_kwargs", None)
        if cleanup_kwargs is not None:
            if not isinstance(cleanup_kwargs, Mapping):
                raise TypeError("cleanup_config_kwargs must be a mapping")
            runner_kwargs["cleanup_config"] = stability_cleanup.StabilityCleanupConfig(
                **dict(cleanup_kwargs)
            )
        return list(
            stability_cleanup.run_track2p_policy_stability_cleanup(
                preset.config,
                **runner_kwargs,
            )
        )
    if preset.runner == "supported-gap-cleanup":
        from bayescatrack.experiments import (
            track2p_policy_supported_gap_component_cleanup as supported_gap_cleanup,
        )

        run_cleanup = (
            supported_gap_cleanup.run_track2p_policy_supported_gap_component_cleanup
        )
        output = run_cleanup(
            preset.config,
            **dict(preset.runner_kwargs or {}),
        )
        return list(output.results)
    if preset.runner == "confidence-ordered-strict-gap-cleanup":
        from bayescatrack.experiments import (
            track2p_policy_confidence_ordered_strict_gap_cleanup as confidence_gap_cleanup,
        )

        runner_kwargs = dict(preset.runner_kwargs or {})
        cleanup_kwargs = runner_kwargs.pop("cleanup_config_kwargs", None)
        if cleanup_kwargs is not None:
            if not isinstance(cleanup_kwargs, Mapping):
                raise TypeError("cleanup_config_kwargs must be a mapping")
            runner_kwargs["cleanup_config"] = (
                confidence_gap_cleanup.ComponentCleanupConfig(**dict(cleanup_kwargs))
            )
        gate_kwargs = runner_kwargs.pop("gate_config_kwargs", None)
        if gate_kwargs is not None:
            if not isinstance(gate_kwargs, Mapping):
                raise TypeError("gate_config_kwargs must be a mapping")
            runner_kwargs["gate_config"] = confidence_gap_cleanup.StrictGapGateConfig(
                **dict(gate_kwargs)
            )
        output = confidence_gap_cleanup.run_track2p_policy_confidence_ordered_strict_gated_gap_cleanup(
            preset.config,
            **runner_kwargs,
        )
        return list(output.results)
    if preset.runner == "teacher-adjacent-rescue":
        from bayescatrack.experiments import (
            track2p_policy_teacher_adjacent_rescue as teacher_rescue,
        )

        runner_kwargs = dict(preset.runner_kwargs or {})
        cleanup_kwargs = runner_kwargs.pop("cleanup_config_kwargs", None)
        if cleanup_kwargs is not None:
            if not isinstance(cleanup_kwargs, Mapping):
                raise TypeError("cleanup_config_kwargs must be a mapping")
            runner_kwargs["cleanup_config"] = teacher_rescue.ComponentCleanupConfig(
                **dict(cleanup_kwargs)
            )
        output = teacher_rescue.run_track2p_policy_teacher_adjacent_rescue(
            preset.config,
            **runner_kwargs,
        )
        return list(output.results)
    raise ValueError(f"Unsupported accuracy preset runner: {preset.runner!r}")


def accuracy_preset_metadata(
    presets: Sequence[AccuracyPreset],
) -> tuple[Mapping[str, object], ...]:
    """Return compact serializable metadata for reporting preset sweeps."""

    rows: list[dict[str, object]] = []
    for preset in presets:
        cfg = preset.config
        rows.append(
            {
                "name": preset.name,
                "description": preset.description,
                "runner": preset.runner,
                "cost": cfg.cost,
                "transform_type": cfg.transform_type,
                "max_gap": cfg.max_gap,
                "cost_threshold": (
                    "none" if cfg.cost_threshold is None else cfg.cost_threshold
                ),
                "candidate_pruning": cfg.candidate_pruning_config is not None,
                "track2p_policy_prior": cfg.track2p_policy_prior_config is not None,
                "dynamic_edge_prior": cfg.dynamic_edge_prior_config is not None,
                "higher_order_consistency": cfg.higher_order_consistency_config
                is not None,
                "consensus_prior": cfg.consensus_prior_config is not None,
                "stability_cleanup": preset.runner == "stability-cleanup",
                "supported_gap_cleanup": preset.runner == "supported-gap-cleanup",
                "confidence_ordered_strict_gap_cleanup": (
                    preset.runner == "confidence-ordered-strict-gap-cleanup"
                ),
                "teacher_adjacent_rescue": preset.runner == "teacher-adjacent-rescue",
            }
        )
    return tuple(rows)


__all__ = (
    "AccuracyPreset",
    "AccuracyPresetName",
    "accuracy_preset_metadata",
    "build_track2p_accuracy_presets",
    "run_track2p_accuracy_presets",
)
