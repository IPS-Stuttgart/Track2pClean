"""Strict validation for assignment-prior sweep scalar costs.

Assignment-prior sweeps can be configured programmatically through
``Track2pBenchmarkConfig``.  Python and NumPy boolean scalars otherwise pass
through ``float()`` as ``1.0`` or ``0.0``, which can silently turn type mistakes
into solver-prior settings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_MARKER = "_bayescatrack_assignment_prior_sweep_validation"


def install_assignment_prior_sweep_validation() -> None:
    """Install idempotent boolean rejection for benchmark sweep values."""

    from bayescatrack.experiments import track2p_benchmark as benchmark

    original_finite_float = benchmark._finite_float  # pylint: disable=protected-access
    if getattr(original_finite_float, _MARKER, False):
        return

    def _finite_float_without_boolean_scalars(value: object, option_name: str) -> float:
        if _is_boolean_scalar(value):
            raise ValueError(f"{option_name} values must be finite numbers")
        return original_finite_float(value, option_name)

    setattr(_finite_float_without_boolean_scalars, _MARKER, True)
    setattr(_finite_float_without_boolean_scalars, "_bayescatrack_original", original_finite_float)
    benchmark._finite_float = _finite_float_without_boolean_scalars  # pylint: disable=protected-access


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if not isinstance(value, np.ndarray):
        return False
    array = np.asarray(value, dtype=object)
    if array.shape == ():
        return isinstance(array.item(), (bool, np.bool_))
    if array.size == 1:
        return isinstance(array.reshape(-1)[0], (bool, np.bool_))
    return False


__all__ = ["install_assignment_prior_sweep_validation"]
