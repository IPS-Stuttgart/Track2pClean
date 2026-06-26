"""Manifest integration for the FullMHT history-consistency runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bayescatrack.experiments import _full_mht_manifest_integration as full_mht_manifest
from bayescatrack.experiments.full_mht_history_consistency_model import (
    IdentityHistoryConsistencyConfig,
)

FULL_MHT_HISTORY_CONSISTENCY_RUNNER = (
    "track2p-policy-full-mht-history-consistency"
)
FULL_MHT_HISTORY_CONSISTENCY_ALIASES = {
    FULL_MHT_HISTORY_CONSISTENCY_RUNNER,
    "track2p-full-mht-history-consistency",
    "track2p-pyrecest-full-mht-history-consistency",
    "track2p-component-full-mht-history-consistency",
}
FULL_MHT_HISTORY_CONSISTENCY_FIELDS = {
    "history_consistency_weight",
    "history_consistency_min_history_edges",
    "history_consistency_min_feature_scale",
    "history_consistency_joint_margin",
    "history_consistency_score_clip",
}


def _runner_fields() -> set[str]:
    return set(full_mht_manifest.FULL_MHT_FIELDS | FULL_MHT_HISTORY_CONSISTENCY_FIELDS)


def install_full_mht_history_consistency_manifest_integration() -> None:
    """Install manifest support for the history-consistency FullMHT runner."""

    from bayescatrack.experiments import benchmark_manifest as manifest
    from bayescatrack.experiments._full_mht_manifest_integration import (
        install_full_mht_manifest_integration,
    )
    from bayescatrack.experiments.full_mht_no_prior_continuation_manifest_integration import (
        install_full_mht_no_prior_continuation_manifest_integration,
    )

    install_full_mht_manifest_integration()
    install_full_mht_no_prior_continuation_manifest_integration()
    if getattr(
        manifest,
        "_bayescatrack_full_mht_history_consistency_manifest_integration",
        False,
    ):
        return

    runner_fields = _runner_fields()
    manifest.RUNNER_SPECIFIC_FIELDS.update(runner_fields)
    manifest.RUN_SPEC_FIELDS.update(runner_fields)
    manifest.RUNNER_CONFIG_FIELDS[FULL_MHT_HISTORY_CONSISTENCY_RUNNER] = set(
        manifest.TRACK2P_CONFIG_FIELDS | runner_fields
    )
    for alias in FULL_MHT_HISTORY_CONSISTENCY_ALIASES:
        manifest.RUNNER_ALIASES[alias] = FULL_MHT_HISTORY_CONSISTENCY_RUNNER
    manifest.RUNNER_CHOICES = frozenset(manifest.RUNNER_ALIASES)

    original_runner_specific_fields = manifest._runner_specific_fields
    original_runner_kwargs = manifest._runner_kwargs
    original_run_config = manifest._run_config
    original_run_benchmark_rows = manifest._run_benchmark_rows
    original_run_manifest_entry = getattr(manifest, "_run_manifest_entry", None)

    def _runner_specific_fields_with_history_consistency(runner: str) -> set[str]:
        if runner == FULL_MHT_HISTORY_CONSISTENCY_RUNNER:
            return _runner_fields()
        return original_runner_specific_fields(runner)

    def _runner_kwargs_with_history_consistency(
        run_data: Mapping[str, Any], runner: str
    ) -> dict[str, Any]:
        if runner == FULL_MHT_HISTORY_CONSISTENCY_RUNNER:
            return {key: run_data[key] for key in _runner_fields() if key in run_data}
        return original_runner_kwargs(run_data, runner)

    def _run_config_with_history_consistency(
        runner: str, run_data: Mapping[str, Any], *, base_dir: Any
    ) -> Any:
        if runner == FULL_MHT_HISTORY_CONSISTENCY_RUNNER:
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

    def _run_benchmark_rows_with_history_consistency(run_spec: Any) -> list[dict[str, Any]]:
        if run_spec.runner == FULL_MHT_HISTORY_CONSISTENCY_RUNNER:
            return _run_track2p_policy_full_mht_history_consistency_rows(
                run_spec.config,
                dict(run_spec.runner_kwargs or {}),
            )
        return original_run_benchmark_rows(run_spec)

    def _run_manifest_entry_with_history_consistency(run_spec: Any) -> list[dict[str, Any]]:
        if run_spec.runner == FULL_MHT_HISTORY_CONSISTENCY_RUNNER:
            return _run_track2p_policy_full_mht_history_consistency_rows(
                run_spec.config,
                dict(run_spec.runner_kwargs or {}),
            )
        if original_run_manifest_entry is None:  # pragma: no cover - legacy guard
            return original_run_benchmark_rows(run_spec)
        return original_run_manifest_entry(run_spec)

    manifest._runner_specific_fields = _runner_specific_fields_with_history_consistency
    manifest._runner_kwargs = _runner_kwargs_with_history_consistency
    manifest._run_config = _run_config_with_history_consistency
    manifest._run_benchmark_rows = _run_benchmark_rows_with_history_consistency
    if original_run_manifest_entry is not None:
        manifest._run_manifest_entry = _run_manifest_entry_with_history_consistency
    manifest._bayescatrack_full_mht_history_consistency_manifest_integration = True


def _run_track2p_policy_full_mht_history_consistency_rows(
    config: Any, options: Mapping[str, Any]
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.track2p_policy_full_mht_history_consistency_benchmark import (
        _patched_full_mht_runner,
    )

    history_config = _history_consistency_config_from_options(options)
    with _patched_full_mht_runner(history_config):
        return full_mht_manifest._run_track2p_policy_full_mht_rows(config, options)


def _history_consistency_config_from_options(
    options: Mapping[str, Any]
) -> IdentityHistoryConsistencyConfig:
    from bayescatrack.experiments import benchmark_manifest as manifest

    defaults = IdentityHistoryConsistencyConfig()
    return IdentityHistoryConsistencyConfig(
        weight=manifest._float_option(
            options,
            "history_consistency_weight",
            default=defaults.weight,
        ),
        min_history_edges=manifest._positive_int_option(
            options,
            "history_consistency_min_history_edges",
            default=defaults.min_history_edges,
        ),
        min_feature_scale=manifest._float_option(
            options,
            "history_consistency_min_feature_scale",
            default=defaults.min_feature_scale,
        ),
        joint_margin=manifest._float_option(
            options,
            "history_consistency_joint_margin",
            default=defaults.joint_margin,
        ),
        score_clip=manifest._float_option(
            options,
            "history_consistency_score_clip",
            default=defaults.score_clip,
        ),
    )
