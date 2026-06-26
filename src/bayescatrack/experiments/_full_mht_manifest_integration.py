"""Manifest integration for the full scan-assignment Track2p MHT runner.

The full-MHT runner is the first Track2p-cleaning experiment that opens a beam
over full identity histories instead of selecting post-hoc residual edits.  This
integration makes the runner executable from JSON benchmark manifests, including
the proposal-prior, prior-veto, calibrated prior-survival, terminal completion,
and history-dynamics controls used by the current method probe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FULL_MHT_RUNNER = "track2p-policy-full-mht"
FULL_MHT_ALIASES = {
    FULL_MHT_RUNNER,
    "track2p-full-mht",
    "track2p-pyrecest-full-mht",
}
FULL_MHT_PRIOR_SURVIVAL_FLOAT_FIELDS = {
    "track2p_prior_survival_weight",
    "track2p_prior_survival_min_anchor_registered_iou",
    "track2p_prior_survival_min_anchor_shifted_iou",
    "track2p_prior_survival_max_anchor_growth_mahalanobis",
    "track2p_prior_survival_max_anchor_growth_residual",
    "track2p_prior_survival_min_anchor_cell_probability",
    "track2p_prior_survival_max_background_registered_iou",
    "track2p_prior_survival_max_background_shifted_iou",
    "track2p_prior_survival_min_background_growth_mahalanobis",
    "track2p_prior_survival_min_background_growth_residual",
    "track2p_prior_survival_max_background_cell_probability",
    "track2p_prior_survival_min_feature_scale",
    "track2p_prior_survival_per_feature_clip",
    "track2p_prior_survival_score_clip",
}
FULL_MHT_PRIOR_SURVIVAL_INT_FIELDS = {
    "track2p_prior_survival_max_anchor_rank",
    "track2p_prior_survival_min_examples_per_class",
}
FULL_MHT_PRIOR_SURVIVAL_FIELDS = (
    FULL_MHT_PRIOR_SURVIVAL_FLOAT_FIELDS | FULL_MHT_PRIOR_SURVIVAL_INT_FIELDS
)
FULL_MHT_TERMINAL_COMPLETION_FLOAT_FIELDS = {
    "terminal_incomplete_history_weight",
}
FULL_MHT_HISTORY_DYNAMICS_FLOAT_FIELDS = {
    "terminal_motion_history_weight",
}
FULL_MHT_FIELDS = {
    "threshold_method",
    "iou_distance_threshold",
    "beam_width",
    "scan_hypotheses",
    "edge_top_k",
    "identity_diverse_beam",
    "miss_cost",
    "full_mht_max_gap",
    "gap_reactivation_cost",
    "min_output_observations",
    "min_edge_score",
    "seed_source",
    "max_seed_tracks",
    "association_score_mode",
    "association_likelihood_weight",
    "association_likelihood_clip",
    "registered_iou_weight",
    "shifted_iou_weight",
    "area_ratio_weight",
    "cell_probability_weight",
    "centroid_distance_weight",
    "threshold_margin_weight",
    "growth_residual_weight",
    "growth_mahalanobis_weight",
    "local_deformation_weight",
    "track2p_prior_weight",
    "track2p_non_prior_penalty",
    "track2p_prior_switch_penalty",
    "track2p_no_prior_successor_penalty",
    "track2p_prior_miss_penalty",
    "track2p_prior_risk_mahalanobis_weight",
    "track2p_prior_risk_mahalanobis_offset",
    "track2p_prior_risk_registered_iou_weight",
    "track2p_prior_risk_registered_iou_floor",
    "track2p_prior_risk_scan_weight",
    "track2p_prior_veto_penalty",
    "track2p_prior_veto_min_growth_residual_mahalanobis",
    "track2p_prior_veto_min_growth_residual",
    "track2p_prior_veto_min_registered_iou",
    "track2p_prior_veto_max_registered_iou",
    "track2p_prior_veto_min_shifted_iou",
    "track2p_prior_veto_max_shifted_iou",
    "track2p_prior_veto_min_cell_probability",
    "track2p_prior_veto_max_min_cell_probability",
    "track2p_prior_veto_max_row_rank",
    "track2p_prior_veto_max_column_rank",
    "track2p_prior_veto_require_terminal_edge",
    "track2p_prior_veto_require_last_session_edge",
    "track2p_prior_veto_require_complete_component",
    "terminal_history_risk_weight",
    "terminal_non_prior_history_weight",
    "terminal_no_prior_successor_history_weight",
    "growth_anchor_min_registered_iou",
    "growth_anchor_min_shifted_iou",
    "growth_anchor_min_cell_probability",
    *FULL_MHT_PRIOR_SURVIVAL_FIELDS,
    *FULL_MHT_TERMINAL_COMPLETION_FLOAT_FIELDS,
    *FULL_MHT_HISTORY_DYNAMICS_FLOAT_FIELDS,
}


def install_full_mht_manifest_integration() -> None:
    """Install manifest support for the full scan-assignment MHT runner."""

    from bayescatrack.experiments import benchmark_manifest as manifest

    if getattr(manifest, "_bayescatrack_full_mht_manifest_integration", False):
        return

    manifest.RUNNER_SPECIFIC_FIELDS.update(FULL_MHT_FIELDS)
    manifest.RUN_SPEC_FIELDS.update(FULL_MHT_FIELDS)
    manifest.RUNNER_CONFIG_FIELDS[FULL_MHT_RUNNER] = set(
        manifest.TRACK2P_CONFIG_FIELDS | FULL_MHT_FIELDS
    )
    for alias in FULL_MHT_ALIASES:
        manifest.RUNNER_ALIASES[alias] = FULL_MHT_RUNNER
    manifest.RUNNER_CHOICES = frozenset(manifest.RUNNER_ALIASES)

    original_runner_specific_fields = manifest._runner_specific_fields
    original_runner_kwargs = manifest._runner_kwargs
    original_run_config = manifest._run_config
    original_run_benchmark_rows = manifest._run_benchmark_rows
    original_run_manifest_entry = getattr(manifest, "_run_manifest_entry", None)

    def _runner_specific_fields_with_full_mht(runner: str) -> set[str]:
        if runner == FULL_MHT_RUNNER:
            return set(FULL_MHT_FIELDS)
        return original_runner_specific_fields(runner)

    def _runner_kwargs_with_full_mht(
        run_data: Mapping[str, Any], runner: str
    ) -> dict[str, Any]:
        if runner == FULL_MHT_RUNNER:
            return {key: run_data[key] for key in FULL_MHT_FIELDS if key in run_data}
        return original_runner_kwargs(run_data, runner)

    def _run_config_with_full_mht(
        runner: str, run_data: Mapping[str, Any], *, base_dir: Any
    ) -> Any:
        if runner == FULL_MHT_RUNNER:
            config_defaults = {
                "method": "global-assignment",
                "include_non_cells": False,
                "weighted_masks": False,
                "weighted_centroids": False,
                "exclude_overlapping_pixels": False,
            }
            config_kwargs = manifest._track2p_config_kwargs(
                run_data,
                base_dir=base_dir,
                config_defaults=config_defaults,
                required=("data",),
            )
            return manifest.Track2pBenchmarkConfig(**config_kwargs)
        return original_run_config(runner, run_data, base_dir=base_dir)

    def _run_benchmark_rows_with_full_mht(run_spec: Any) -> list[dict[str, Any]]:
        if run_spec.runner == FULL_MHT_RUNNER:
            return _run_track2p_policy_full_mht_rows(
                run_spec.config, dict(run_spec.runner_kwargs or {})
            )
        return original_run_benchmark_rows(run_spec)

    def _run_manifest_entry_with_full_mht(run_spec: Any) -> list[dict[str, Any]]:
        if run_spec.runner == FULL_MHT_RUNNER:
            return _run_track2p_policy_full_mht_rows(
                run_spec.config, dict(run_spec.runner_kwargs or {})
            )
        if original_run_manifest_entry is None:  # pragma: no cover - legacy guard
            return original_run_benchmark_rows(run_spec)
        return original_run_manifest_entry(run_spec)

    manifest._runner_specific_fields = _runner_specific_fields_with_full_mht
    manifest._runner_kwargs = _runner_kwargs_with_full_mht
    manifest._run_config = _run_config_with_full_mht
    manifest._run_benchmark_rows = _run_benchmark_rows_with_full_mht
    if original_run_manifest_entry is not None:
        manifest._run_manifest_entry = _run_manifest_entry_with_full_mht
    manifest._bayescatrack_full_mht_manifest_integration = True


def _run_track2p_policy_full_mht_rows(
    config: Any, options: Mapping[str, Any]
) -> list[dict[str, Any]]:
    from bayescatrack.experiments import benchmark_manifest as manifest
    from bayescatrack.experiments.track2p_policy_benchmark import (
        TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    from bayescatrack.experiments.track2p_policy_full_mht_benchmark import (
        run_track2p_policy_full_mht,
    )

    if _uses_prior_survival(options):
        from bayescatrack.experiments.full_mht_prior_survival_integration import (
            install_full_mht_prior_survival_scoring,
        )

        install_full_mht_prior_survival_scoring()
    if _uses_terminal_completion(options):
        from bayescatrack.experiments.full_mht_terminal_completion_integration import (
            install_full_mht_terminal_completion_objective,
        )

        install_full_mht_terminal_completion_objective()
    if _uses_history_dynamics(options):
        from bayescatrack.experiments.full_mht_history_dynamics_integration import (
            install_full_mht_history_dynamics_objective,
        )

        install_full_mht_history_dynamics_objective()

    output = run_track2p_policy_full_mht(
        config,
        threshold_method=manifest._policy_threshold_method(
            options.get("threshold_method", TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD)
        ),
        iou_distance_threshold=manifest._float_option(
            options,
            "iou_distance_threshold",
            default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        ),
        transform_type=config.transform_type,
        cell_probability_threshold=config.cell_probability_threshold,
        mht_config=_full_mht_config_from_options(options),
        progress=bool(getattr(config, "progress", False)),
    )
    return [result.to_dict() for result in output.results]


def _uses_prior_survival(options: Mapping[str, Any]) -> bool:
    return any(key in options for key in FULL_MHT_PRIOR_SURVIVAL_FIELDS)


def _uses_terminal_completion(options: Mapping[str, Any]) -> bool:
    return any(key in options for key in FULL_MHT_TERMINAL_COMPLETION_FLOAT_FIELDS)


def _uses_history_dynamics(options: Mapping[str, Any]) -> bool:
    return any(key in options for key in FULL_MHT_HISTORY_DYNAMICS_FLOAT_FIELDS)


def _full_mht_config_from_options(options: Mapping[str, Any]) -> Any:
    from bayescatrack.experiments import benchmark_manifest as manifest
    from bayescatrack.experiments.track2p_policy_full_mht_benchmark import (
        FullMHTConfig,
    )

    defaults = FullMHTConfig()
    config = FullMHTConfig(
        beam_width=manifest._positive_int_option(
            options, "beam_width", default=defaults.beam_width
        ),
        scan_hypotheses=manifest._positive_int_option(
            options, "scan_hypotheses", default=defaults.scan_hypotheses
        ),
        edge_top_k=manifest._positive_int_option(
            options, "edge_top_k", default=defaults.edge_top_k
        ),
        identity_diverse_beam=manifest._bool_option(
            options, "identity_diverse_beam", default=defaults.identity_diverse_beam
        ),
        miss_cost=manifest._float_option(options, "miss_cost", default=defaults.miss_cost),
        max_gap=manifest._nonnegative_int_option(
            options, "full_mht_max_gap", default=defaults.max_gap
        ),
        gap_reactivation_cost=manifest._float_option(
            options, "gap_reactivation_cost", default=defaults.gap_reactivation_cost
        ),
        min_output_observations=manifest._positive_int_option(
            options,
            "min_output_observations",
            default=defaults.min_output_observations,
        ),
        min_edge_score=manifest._float_option(
            options, "min_edge_score", default=defaults.min_edge_score
        ),
        seed_source=_seed_source(options, defaults.seed_source),
        max_seed_tracks=manifest._positive_int_or_none(
            options.get("max_seed_tracks", defaults.max_seed_tracks),
            name="max_seed_tracks",
        ),
        association_score_mode=_association_score_mode(
            options, defaults.association_score_mode
        ),
        association_likelihood_weight=manifest._float_option(
            options,
            "association_likelihood_weight",
            default=defaults.association_likelihood_weight,
        ),
        association_likelihood_clip=manifest._float_option(
            options,
            "association_likelihood_clip",
            default=defaults.association_likelihood_clip,
        ),
        registered_iou_weight=manifest._float_option(
            options, "registered_iou_weight", default=defaults.registered_iou_weight
        ),
        shifted_iou_weight=manifest._float_option(
            options, "shifted_iou_weight", default=defaults.shifted_iou_weight
        ),
        area_ratio_weight=manifest._float_option(
            options, "area_ratio_weight", default=defaults.area_ratio_weight
        ),
        cell_probability_weight=manifest._float_option(
            options,
            "cell_probability_weight",
            default=defaults.cell_probability_weight,
        ),
        centroid_distance_weight=manifest._float_option(
            options,
            "centroid_distance_weight",
            default=defaults.centroid_distance_weight,
        ),
        threshold_margin_weight=manifest._float_option(
            options,
            "threshold_margin_weight",
            default=defaults.threshold_margin_weight,
        ),
        growth_residual_weight=manifest._float_option(
            options,
            "growth_residual_weight",
            default=defaults.growth_residual_weight,
        ),
        growth_mahalanobis_weight=manifest._float_option(
            options,
            "growth_mahalanobis_weight",
            default=defaults.growth_mahalanobis_weight,
        ),
        local_deformation_weight=manifest._float_option(
            options,
            "local_deformation_weight",
            default=defaults.local_deformation_weight,
        ),
        track2p_prior_weight=manifest._float_option(
            options, "track2p_prior_weight", default=defaults.track2p_prior_weight
        ),
        track2p_non_prior_penalty=manifest._float_option(
            options,
            "track2p_non_prior_penalty",
            default=defaults.track2p_non_prior_penalty,
        ),
        track2p_prior_switch_penalty=manifest._float_option(
            options,
            "track2p_prior_switch_penalty",
            default=defaults.track2p_prior_switch_penalty,
        ),
        track2p_no_prior_successor_penalty=manifest._float_option(
            options,
            "track2p_no_prior_successor_penalty",
            default=defaults.track2p_no_prior_successor_penalty,
        ),
        track2p_prior_miss_penalty=manifest._float_option(
            options,
            "track2p_prior_miss_penalty",
            default=defaults.track2p_prior_miss_penalty,
        ),
        track2p_prior_risk_mahalanobis_weight=manifest._float_option(
            options,
            "track2p_prior_risk_mahalanobis_weight",
            default=defaults.track2p_prior_risk_mahalanobis_weight,
        ),
        track2p_prior_risk_mahalanobis_offset=manifest._float_option(
            options,
            "track2p_prior_risk_mahalanobis_offset",
            default=defaults.track2p_prior_risk_mahalanobis_offset,
        ),
        track2p_prior_risk_registered_iou_weight=manifest._float_option(
            options,
            "track2p_prior_risk_registered_iou_weight",
            default=defaults.track2p_prior_risk_registered_iou_weight,
        ),
        track2p_prior_risk_registered_iou_floor=manifest._float_option(
            options,
            "track2p_prior_risk_registered_iou_floor",
            default=defaults.track2p_prior_risk_registered_iou_floor,
        ),
        track2p_prior_risk_scan_weight=manifest._float_option(
            options,
            "track2p_prior_risk_scan_weight",
            default=defaults.track2p_prior_risk_scan_weight,
        ),
        track2p_prior_veto_penalty=manifest._float_option(
            options,
            "track2p_prior_veto_penalty",
            default=defaults.track2p_prior_veto_penalty,
        ),
        track2p_prior_veto_min_growth_residual_mahalanobis=manifest._float_option(
            options,
            "track2p_prior_veto_min_growth_residual_mahalanobis",
            default=defaults.track2p_prior_veto_min_growth_residual_mahalanobis,
        ),
        track2p_prior_veto_min_growth_residual=manifest._float_option(
            options,
            "track2p_prior_veto_min_growth_residual",
            default=defaults.track2p_prior_veto_min_growth_residual,
        ),
        track2p_prior_veto_min_registered_iou=manifest._float_option(
            options,
            "track2p_prior_veto_min_registered_iou",
            default=defaults.track2p_prior_veto_min_registered_iou,
        ),
        track2p_prior_veto_max_registered_iou=_optional_float_with_default(
            options,
            "track2p_prior_veto_max_registered_iou",
            default=defaults.track2p_prior_veto_max_registered_iou,
        ),
        track2p_prior_veto_min_shifted_iou=manifest._float_option(
            options,
            "track2p_prior_veto_min_shifted_iou",
            default=defaults.track2p_prior_veto_min_shifted_iou,
        ),
        track2p_prior_veto_max_shifted_iou=_optional_float_with_default(
            options,
            "track2p_prior_veto_max_shifted_iou",
            default=defaults.track2p_prior_veto_max_shifted_iou,
        ),
        track2p_prior_veto_min_cell_probability=manifest._float_option(
            options,
            "track2p_prior_veto_min_cell_probability",
            default=defaults.track2p_prior_veto_min_cell_probability,
        ),
        track2p_prior_veto_max_min_cell_probability=_optional_float_with_default(
            options,
            "track2p_prior_veto_max_min_cell_probability",
            default=defaults.track2p_prior_veto_max_min_cell_probability,
        ),
        track2p_prior_veto_max_row_rank=manifest._positive_int_option(
            options,
            "track2p_prior_veto_max_row_rank",
            default=defaults.track2p_prior_veto_max_row_rank,
        ),
        track2p_prior_veto_max_column_rank=manifest._positive_int_option(
            options,
            "track2p_prior_veto_max_column_rank",
            default=defaults.track2p_prior_veto_max_column_rank,
        ),
        track2p_prior_veto_require_terminal_edge=manifest._bool_option(
            options,
            "track2p_prior_veto_require_terminal_edge",
            default=defaults.track2p_prior_veto_require_terminal_edge,
        ),
        track2p_prior_veto_require_last_session_edge=manifest._bool_option(
            options,
            "track2p_prior_veto_require_last_session_edge",
            default=defaults.track2p_prior_veto_require_last_session_edge,
        ),
        track2p_prior_veto_require_complete_component=manifest._bool_option(
            options,
            "track2p_prior_veto_require_complete_component",
            default=defaults.track2p_prior_veto_require_complete_component,
        ),
        terminal_history_risk_weight=manifest._float_option(
            options,
            "terminal_history_risk_weight",
            default=defaults.terminal_history_risk_weight,
        ),
        terminal_non_prior_history_weight=manifest._float_option(
            options,
            "terminal_non_prior_history_weight",
            default=defaults.terminal_non_prior_history_weight,
        ),
        terminal_no_prior_successor_history_weight=manifest._float_option(
            options,
            "terminal_no_prior_successor_history_weight",
            default=defaults.terminal_no_prior_successor_history_weight,
        ),
        growth_anchor_min_registered_iou=manifest._float_option(
            options,
            "growth_anchor_min_registered_iou",
            default=defaults.growth_anchor_min_registered_iou,
        ),
        growth_anchor_min_shifted_iou=manifest._float_option(
            options,
            "growth_anchor_min_shifted_iou",
            default=defaults.growth_anchor_min_shifted_iou,
        ),
        growth_anchor_min_cell_probability=manifest._float_option(
            options,
            "growth_anchor_min_cell_probability",
            default=defaults.growth_anchor_min_cell_probability,
        ),
    )
    config = _attach_prior_survival_options(config, options)
    config = _attach_terminal_completion_options(config, options)
    return _attach_history_dynamics_options(config, options)


def _attach_prior_survival_options(config: Any, options: Mapping[str, Any]) -> Any:
    from bayescatrack.experiments import benchmark_manifest as manifest

    for key in sorted(FULL_MHT_PRIOR_SURVIVAL_FLOAT_FIELDS):
        if key in options:
            object.__setattr__(
                config,
                key,
                manifest._float_option(options, key, default=0.0),
            )
    for key in sorted(FULL_MHT_PRIOR_SURVIVAL_INT_FIELDS):
        if key in options:
            object.__setattr__(
                config,
                key,
                manifest._positive_int_option(options, key, default=1),
            )
    return config


def _attach_terminal_completion_options(config: Any, options: Mapping[str, Any]) -> Any:
    from bayescatrack.experiments import benchmark_manifest as manifest

    for key in sorted(FULL_MHT_TERMINAL_COMPLETION_FLOAT_FIELDS):
        if key in options:
            object.__setattr__(
                config,
                key,
                manifest._float_option(options, key, default=0.0),
            )
    return config


def _attach_history_dynamics_options(config: Any, options: Mapping[str, Any]) -> Any:
    from bayescatrack.experiments import benchmark_manifest as manifest

    for key in sorted(FULL_MHT_HISTORY_DYNAMICS_FLOAT_FIELDS):
        if key in options:
            object.__setattr__(
                config,
                key,
                manifest._float_option(options, key, default=0.0),
            )
    return config


def _optional_float_with_default(
    options: Mapping[str, Any], key: str, *, default: float | None
) -> float | None:
    if key not in options:
        return default
    from bayescatrack.experiments import benchmark_manifest as manifest

    return manifest._optional_float_option(options, key)


def _seed_source(options: Mapping[str, Any], default: str) -> str:
    value = str(options.get("seed_source", default))
    if value not in {"reference", "all-cells", "track2p-output"}:
        raise ValueError(
            "seed_source must be 'reference', 'all-cells', or 'track2p-output'"
        )
    return value


def _association_score_mode(options: Mapping[str, Any], default: str) -> str:
    value = str(options.get("association_score_mode", default))
    if value not in {"heuristic", "calibrated-likelihood"}:
        raise ValueError(
            "association_score_mode must be 'heuristic' or 'calibrated-likelihood'"
        )
    return value
