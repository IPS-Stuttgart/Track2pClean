"""Validation for configurable LOSO calibration feature CLI options.

The configurable LOSO CLI treats ``--calibration-features`` and repeated
``--calibration-feature`` as explicit overrides of the named feature-set preset.
Empty explicit values are therefore operator mistakes and must not silently fall
back to the default preset.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

_PATCH_MARKER = "_bayescatrack_configurable_loso_feature_cli_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_configurable_loso_feature_cli_original"


def install_configurable_loso_feature_cli_validation() -> None:
    """Install idempotent strict validation for configurable LOSO feature lists."""

    from . import track2p_configurable_loso_calibration as _target

    current = _target._resolved_calibration_feature_names
    if getattr(current, _PATCH_MARKER, False):
        return
    original = getattr(current, _ORIGINAL_ATTR, current)

    def _resolved_calibration_feature_names(args: argparse.Namespace) -> tuple[str, ...]:
        names: list[str] = []
        if args.calibration_features is not None:
            names.extend(
                _parse_comma_separated_features(
                    args.calibration_features,
                    option_name="--calibration-features",
                )
            )
        if args.calibration_feature is not None:
            names.extend(
                _parse_repeated_features(
                    args.calibration_feature,
                    option_name="--calibration-feature",
                )
            )
        if names:
            return tuple(dict.fromkeys(names))
        return original(args)

    setattr(_resolved_calibration_feature_names, _PATCH_MARKER, True)
    setattr(_resolved_calibration_feature_names, _ORIGINAL_ATTR, original)
    _target._resolved_calibration_feature_names = _resolved_calibration_feature_names


def _parse_comma_separated_features(raw_value: Any, *, option_name: str) -> tuple[str, ...]:
    if not isinstance(raw_value, str):
        raise ValueError(f"{option_name} must be a comma-separated list of feature names")
    tokens = tuple(token.strip() for token in raw_value.split(","))
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{option_name} must be a comma-separated list of feature names")
    return tokens


def _parse_repeated_features(values: Sequence[Any], *, option_name: str) -> tuple[str, ...]:
    tokens = tuple(str(value).strip() for value in values)
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{option_name} must contain non-empty feature names")
    return tokens
