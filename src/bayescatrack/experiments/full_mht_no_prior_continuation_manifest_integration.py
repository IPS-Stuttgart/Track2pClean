"""Manifest support for the FullMHT no-prior continuation likelihood hook."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

NO_PRIOR_CONTINUATION_FLOAT_FIELDS = {
    "no_prior_continuation_likelihood_weight",
    "no_prior_continuation_min_anchor_registered_iou",
    "no_prior_continuation_min_anchor_shifted_iou",
    "no_prior_continuation_max_anchor_growth_mahalanobis",
    "no_prior_continuation_max_anchor_growth_residual",
    "no_prior_continuation_min_anchor_cell_probability",
    "no_prior_continuation_max_anchor_local_deformation",
    "no_prior_continuation_max_background_registered_iou",
    "no_prior_continuation_max_background_shifted_iou",
    "no_prior_continuation_min_background_growth_mahalanobis",
    "no_prior_continuation_min_background_growth_residual",
    "no_prior_continuation_max_background_cell_probability",
    "no_prior_continuation_min_background_local_deformation",
    "no_prior_continuation_min_feature_scale",
    "no_prior_continuation_per_feature_clip",
    "no_prior_continuation_score_clip",
}
NO_PRIOR_CONTINUATION_INT_FIELDS = {
    "no_prior_continuation_max_anchor_rank",
    "no_prior_continuation_min_examples_per_class",
}
NO_PRIOR_CONTINUATION_FIELDS = (
    NO_PRIOR_CONTINUATION_FLOAT_FIELDS | NO_PRIOR_CONTINUATION_INT_FIELDS
)


def install_full_mht_no_prior_continuation_manifest_integration() -> None:
    """Install manifest fields and scoring-hook activation for no-prior likelihood."""

    from bayescatrack.experiments import _full_mht_manifest_integration as full_mht_manifest
    from bayescatrack.experiments import benchmark_manifest as manifest

    if getattr(manifest, "_bayescatrack_no_prior_continuation_manifest", False):
        return

    full_mht_manifest.FULL_MHT_FIELDS.update(NO_PRIOR_CONTINUATION_FIELDS)
    manifest.RUNNER_SPECIFIC_FIELDS.update(NO_PRIOR_CONTINUATION_FIELDS)
    manifest.RUN_SPEC_FIELDS.update(NO_PRIOR_CONTINUATION_FIELDS)
    manifest.RUNNER_CONFIG_FIELDS[full_mht_manifest.FULL_MHT_RUNNER].update(
        NO_PRIOR_CONTINUATION_FIELDS
    )

    original_config_from_options = full_mht_manifest._full_mht_config_from_options
    original_run_rows = full_mht_manifest._run_track2p_policy_full_mht_rows

    def _full_mht_config_from_options_with_no_prior_continuation(
        options: Mapping[str, Any]
    ) -> Any:
        config = original_config_from_options(options)
        return _attach_no_prior_continuation_options(config, options)

    def _run_track2p_policy_full_mht_rows_with_no_prior_continuation(
        config: Any, options: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        if _uses_no_prior_continuation(options):
            from bayescatrack.experiments.full_mht_no_prior_continuation_integration import (
                install_full_mht_no_prior_continuation_scoring,
            )

            install_full_mht_no_prior_continuation_scoring()
        return original_run_rows(config, options)

    full_mht_manifest._full_mht_config_from_options = (
        _full_mht_config_from_options_with_no_prior_continuation
    )
    full_mht_manifest._run_track2p_policy_full_mht_rows = (
        _run_track2p_policy_full_mht_rows_with_no_prior_continuation
    )
    manifest._bayescatrack_no_prior_continuation_manifest = True


def _uses_no_prior_continuation(options: Mapping[str, Any]) -> bool:
    return any(key in options for key in NO_PRIOR_CONTINUATION_FIELDS)


def _attach_no_prior_continuation_options(
    config: Any, options: Mapping[str, Any]
) -> Any:
    from bayescatrack.experiments import benchmark_manifest as manifest

    for key in sorted(NO_PRIOR_CONTINUATION_FLOAT_FIELDS):
        if key in options:
            object.__setattr__(
                config,
                key,
                manifest._float_option(options, key, default=0.0),
            )
    for key in sorted(NO_PRIOR_CONTINUATION_INT_FIELDS):
        if key in options:
            object.__setattr__(
                config,
                key,
                manifest._positive_int_option(options, key, default=1),
            )
    return config
