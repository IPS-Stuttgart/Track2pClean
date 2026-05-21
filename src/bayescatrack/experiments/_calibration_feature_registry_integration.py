"""Wire the shared calibration feature registry into existing benchmark CLIs."""

from __future__ import annotations

from typing import Any

from bayescatrack.experiments import calibration_feature_sets as registry


def install_calibration_feature_registry_integration() -> None:
    """Install idempotent compatibility patches for calibration feature presets.

    Older benchmark modules keep local feature-set constants for CLI choices and
    training helpers.  This integration points those public names at the shared
    registry added for benchmark hardening, so documented presets such as
    ``rich`` and ``split-roi`` are accepted consistently without rewriting the
    large benchmark modules in place.
    """

    from bayescatrack.experiments import track2p_benchmark
    from bayescatrack.experiments import track2p_loso_calibration

    _patch_loso_module(track2p_loso_calibration)
    _patch_track2p_benchmark_parser(track2p_benchmark)


def _patch_loso_module(module: Any) -> None:
    if getattr(module, "_bayescatrack_shared_feature_registry", False):
        return
    module.CALIBRATION_FEATURE_SET_CHOICES = registry.CALIBRATION_FEATURE_SET_CHOICES
    module.calibration_feature_names = registry.calibration_feature_names
    module.pairwise_cost_kwargs_for_calibration_features = (
        registry.pairwise_cost_kwargs_for_calibration_features
    )
    module._pairwise_kwargs_request_local_evidence = (  # pylint: disable=protected-access
        registry.pairwise_kwargs_request_local_evidence
    )
    module._uses_local_evidence_features = registry.uses_local_evidence_features  # pylint: disable=protected-access
    module._uses_shifted_overlap_features = registry.uses_shifted_overlap_features  # pylint: disable=protected-access
    module._bayescatrack_shared_feature_registry = True


def _patch_track2p_benchmark_parser(module: Any) -> None:
    original = module.build_arg_parser
    if getattr(original, "_bayescatrack_shared_feature_registry", False):
        return

    def build_arg_parser_with_shared_feature_registry(*args: Any, **kwargs: Any) -> Any:
        parser = original(*args, **kwargs)
        _set_calibration_feature_choices(parser)
        return parser

    build_arg_parser_with_shared_feature_registry._bayescatrack_shared_feature_registry = True  # type: ignore[attr-defined]
    build_arg_parser_with_shared_feature_registry._bayescatrack_original = original  # type: ignore[attr-defined]
    module.build_arg_parser = build_arg_parser_with_shared_feature_registry


def _set_calibration_feature_choices(parser: Any) -> None:
    for action in getattr(parser, "_actions", ()):  # argparse keeps actions here.
        if getattr(action, "dest", None) == "calibration_feature_set":
            action.choices = registry.CALIBRATION_FEATURE_SET_CHOICES
            break
