"""FullMHT runner with calibrated prior-edge survival scoring enabled.

This module is a thin paper-facing command wrapper around
``track2p_policy_full_mht_benchmark``.  The base FullMHT runner remains the single
implementation of scan-assignment tracking; this wrapper installs the calibrated
prior-edge survival scoring layer and exposes its label-free calibration knobs as
normal command-line options.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from bayescatrack.experiments.full_mht_prior_survival_model import (
    PriorEdgeSurvivalConfig,
)

METHOD = "track2p-policy-full-mht-prior-survival"

_SURVIVAL_FLOAT_OPTIONS: tuple[tuple[str, str, float, str], ...] = (
    (
        "track2p_prior_survival_weight",
        "--track2p-prior-survival-weight",
        1.0,
        "Weight for the calibrated prior-edge survival log-likelihood ratio.",
    ),
    (
        "track2p_prior_survival_min_anchor_registered_iou",
        "--track2p-prior-survival-min-anchor-registered-iou",
        PriorEdgeSurvivalConfig.min_anchor_registered_iou,
        "Minimum registered IoU for pseudo-survival anchors.",
    ),
    (
        "track2p_prior_survival_min_anchor_shifted_iou",
        "--track2p-prior-survival-min-anchor-shifted-iou",
        PriorEdgeSurvivalConfig.min_anchor_shifted_iou,
        "Minimum shifted IoU for pseudo-survival anchors.",
    ),
    (
        "track2p_prior_survival_max_anchor_growth_mahalanobis",
        "--track2p-prior-survival-max-anchor-growth-mahalanobis",
        PriorEdgeSurvivalConfig.max_anchor_growth_mahalanobis,
        "Maximum growth Mahalanobis residual for pseudo-survival anchors.",
    ),
    (
        "track2p_prior_survival_max_anchor_growth_residual",
        "--track2p-prior-survival-max-anchor-growth-residual",
        PriorEdgeSurvivalConfig.max_anchor_growth_residual,
        "Maximum growth residual for pseudo-survival anchors.",
    ),
    (
        "track2p_prior_survival_min_anchor_cell_probability",
        "--track2p-prior-survival-min-anchor-cell-probability",
        PriorEdgeSurvivalConfig.min_anchor_cell_probability,
        "Minimum endpoint cell probability for pseudo-survival anchors.",
    ),
    (
        "track2p_prior_survival_max_background_registered_iou",
        "--track2p-prior-survival-max-background-registered-iou",
        PriorEdgeSurvivalConfig.max_background_registered_iou,
        "Maximum registered IoU for pseudo-hazard background examples.",
    ),
    (
        "track2p_prior_survival_max_background_shifted_iou",
        "--track2p-prior-survival-max-background-shifted-iou",
        PriorEdgeSurvivalConfig.max_background_shifted_iou,
        "Maximum shifted IoU for pseudo-hazard background examples.",
    ),
    (
        "track2p_prior_survival_min_background_growth_mahalanobis",
        "--track2p-prior-survival-min-background-growth-mahalanobis",
        PriorEdgeSurvivalConfig.min_background_growth_mahalanobis,
        "Minimum growth Mahalanobis residual for pseudo-hazard background examples.",
    ),
    (
        "track2p_prior_survival_min_background_growth_residual",
        "--track2p-prior-survival-min-background-growth-residual",
        PriorEdgeSurvivalConfig.min_background_growth_residual,
        "Minimum growth residual for pseudo-hazard background examples.",
    ),
    (
        "track2p_prior_survival_max_background_cell_probability",
        "--track2p-prior-survival-max-background-cell-probability",
        PriorEdgeSurvivalConfig.max_background_cell_probability,
        "Maximum endpoint cell probability for pseudo-hazard background examples.",
    ),
    (
        "track2p_prior_survival_min_feature_scale",
        "--track2p-prior-survival-min-feature-scale",
        PriorEdgeSurvivalConfig.min_feature_scale,
        "Minimum robust feature scale used by the survival likelihood model.",
    ),
    (
        "track2p_prior_survival_per_feature_clip",
        "--track2p-prior-survival-per-feature-clip",
        PriorEdgeSurvivalConfig.per_feature_clip,
        "Per-feature survival log-likelihood clip.",
    ),
    (
        "track2p_prior_survival_score_clip",
        "--track2p-prior-survival-score-clip",
        PriorEdgeSurvivalConfig.score_clip,
        "Final survival log-likelihood ratio clip.",
    ),
)

_SURVIVAL_INT_OPTIONS: tuple[tuple[str, str, int, str], ...] = (
    (
        "track2p_prior_survival_max_anchor_rank",
        "--track2p-prior-survival-max-anchor-rank",
        PriorEdgeSurvivalConfig.max_anchor_rank,
        "Maximum row/column rank for pseudo-survival anchors.",
    ),
    (
        "track2p_prior_survival_min_examples_per_class",
        "--track2p-prior-survival-min-examples-per-class",
        PriorEdgeSurvivalConfig.min_examples_per_class,
        "Minimum pseudo examples per class required to enable survival scoring.",
    ),
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the prior-survival FullMHT benchmark parser."""

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    parser = full_mht.build_arg_parser()
    parser.prog = f"bayescatrack benchmark {METHOD}"
    parser.description = (
        "Run FullMHT with calibrated label-free Track2p prior-edge survival scoring."
    )
    _add_prior_survival_arguments(parser)
    return parser


def _survival_only_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    _add_prior_survival_arguments(parser)
    return parser


def _add_prior_survival_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("calibrated prior-edge survival")
    for dest, flag, default, help_text in _SURVIVAL_FLOAT_OPTIONS:
        group.add_argument(flag, dest=dest, type=float, default=default, help=help_text)
    for dest, flag, default, help_text in _SURVIVAL_INT_OPTIONS:
        group.add_argument(flag, dest=dest, type=int, default=default, help=help_text)


def _split_survival_args(argv: Sequence[str]) -> tuple[list[str], dict[str, float | int]]:
    """Return base FullMHT argv plus dynamic survival config attributes."""

    namespace, base_argv = _survival_only_parser().parse_known_args(list(argv))
    attrs: dict[str, float | int] = {}
    for dest, _flag, _default, _help in _SURVIVAL_FLOAT_OPTIONS:
        attrs[dest] = float(getattr(namespace, dest))
    for dest, _flag, _default, _help in _SURVIVAL_INT_OPTIONS:
        attrs[dest] = int(getattr(namespace, dest))
    return list(base_argv), attrs


def _attach_survival_attrs(config: Any, attrs: dict[str, float | int]) -> Any:
    for key, value in attrs.items():
        object.__setattr__(config, key, value)
    return config


def main(argv: list[str] | None = None) -> int:
    """Run FullMHT with calibrated prior-edge survival scoring enabled."""

    raw_argv = list(argv or [])
    if any(arg in {"-h", "--help"} for arg in raw_argv):
        build_arg_parser().parse_args(raw_argv)
        return 0

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht
    from bayescatrack.experiments.full_mht_prior_survival_integration import (
        install_full_mht_prior_survival_scoring,
    )

    base_argv, survival_attrs = _split_survival_args(raw_argv)
    install_full_mht_prior_survival_scoring()

    original_config_class = full_mht.FullMHTConfig

    def full_mht_config_with_prior_survival(*args: Any, **kwargs: Any) -> Any:
        return _attach_survival_attrs(
            original_config_class(*args, **kwargs), survival_attrs
        )

    full_mht.FullMHTConfig = full_mht_config_with_prior_survival
    try:
        return int(full_mht.main(base_argv))
    finally:
        full_mht.FullMHTConfig = original_config_class


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
