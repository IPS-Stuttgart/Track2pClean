"""Strict validation for mask-input sweep option sequences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_PATCH_ATTR = "_track2pclean_mask_input_sweep_option_validation"


def install_mask_input_sweep_option_validation() -> None:
    """Install idempotent fail-fast validation for public sweep option tuples."""

    from . import track2p_mask_input_sweep as sweep

    original_normalise = (
        sweep._normalise_bool_options
    )  # pylint: disable=protected-access
    if getattr(original_normalise, _PATCH_ATTR, False):
        return

    def _normalise_bool_options(
        values: Sequence[bool], *, name: str
    ) -> tuple[bool, ...]:
        normalised_values: list[bool] = []
        for value in values:
            if type(value) is not bool and not isinstance(value, np.bool_):
                raise ValueError(f"{name} options must be booleans")
            normalised_values.append(bool(value))
        normalised = tuple(dict.fromkeys(normalised_values))
        if not normalised:
            raise ValueError(f"At least one {name} option is required")
        return normalised

    setattr(_normalise_bool_options, _PATCH_ATTR, True)
    setattr(_normalise_bool_options, "_track2pclean_original", original_normalise)
    sweep._normalise_bool_options = (
        _normalise_bool_options  # pylint: disable=protected-access
    )


__all__ = ["install_mask_input_sweep_option_validation"]
