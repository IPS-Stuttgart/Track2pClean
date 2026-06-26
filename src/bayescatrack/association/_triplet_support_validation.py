"""Strict validation for triplet-support consistency controls.

Triplet-support skip-edge penalties use numeric controls for the prior strength
and the intermediate-path top-k search. Python booleans are integers, so values
such as ``support_top_k=True`` or ``triplet_weight=True`` could silently alter the
prior instead of being rejected as malformed configuration.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

from . import pyrecest_global_assignment as _global_assignment

_PATCH_ATTR = "_bayescatrack_triplet_support_validation_patch"


def install_triplet_support_validation() -> None:
    """Install an idempotent validator around triplet-support penalties."""

    original = _global_assignment.apply_triplet_support_consistency
    if getattr(original, _PATCH_ATTR, False):
        return

    @wraps(original)
    def apply_triplet_support_consistency_with_validation(
        pairwise_costs: Any,
        *,
        config: Any,
    ) -> dict[tuple[int, int], np.ndarray]:
        return original(
            pairwise_costs,
            config=_normalize_triplet_support_config(config),
        )

    setattr(apply_triplet_support_consistency_with_validation, _PATCH_ATTR, True)
    setattr(
        apply_triplet_support_consistency_with_validation,
        "_bayescatrack_original",
        original,
    )
    _global_assignment.apply_triplet_support_consistency = (  # type: ignore[assignment]
        apply_triplet_support_consistency_with_validation
    )


def _normalize_triplet_support_config(config: Any) -> Any:
    if config is None:
        return None

    try:
        triplet_weight = getattr(config, "triplet_weight")
        support_top_k = getattr(config, "support_top_k")
        support_cost_cap = getattr(config, "support_cost_cap")
        max_penalty = getattr(config, "max_penalty")
    except AttributeError as exc:
        raise ValueError(
            "config must provide triplet-support consistency fields"
        ) from exc

    return _global_assignment.TripletSupportConsistencyConfig(
        triplet_weight=_finite_nonnegative_float(
            triplet_weight,
            name="triplet_weight",
        ),
        support_top_k=_positive_integer_like(
            support_top_k,
            name="support_top_k",
        ),
        support_cost_cap=_optional_finite_nonnegative_float(
            support_cost_cap,
            name="support_cost_cap",
        ),
        max_penalty=_optional_finite_nonnegative_float(
            max_penalty,
            name="max_penalty",
        ),
    )


def _optional_finite_nonnegative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative_float(value, name=name)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative value") from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return numeric_value


def _positive_integer_like(value: Any, *, name: str) -> int:
    integer_value = _integer_like(value, name=name)
    if integer_value < 1:
        raise ValueError(f"{name} must be at least 1")
    return integer_value


def _integer_like(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError:
        pass

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must be an integer")
        try:
            numeric_value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    elif isinstance(value, (float, np.floating)):
        numeric_value = float(value)
    else:
        raise ValueError(f"{name} must be an integer")

    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric_value)


__all__ = ["install_triplet_support_validation"]
