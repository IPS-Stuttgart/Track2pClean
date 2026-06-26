"""Runtime validation for core geometry and pairwise-cost scalar controls."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

_CORE_SCALAR_VALIDATION_INSTALLED_ATTR = "_bayescatrack_core_scalar_validation_installed"
_PATCH_MARKER = "_bayescatrack_core_scalar_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"

_NONNEGATIVE_PAIRWISE_CONTROLS = (
    "centroid_weight",
    "iou_weight",
    "mask_cosine_weight",
    "area_weight",
    "roi_feature_weight",
    "cell_probability_weight",
)
_STRICTLY_POSITIVE_PAIRWISE_CONTROLS = (
    "large_cost",
    "similarity_epsilon",
)
_OPTIONAL_STRICTLY_POSITIVE_PAIRWISE_CONTROLS = (
    "centroid_scale",
    "max_centroid_distance",
)


def install_core_scalar_validation_patches(calcium_plane_cls: type[Any]) -> None:
    """Install idempotent validation wrappers for core scalar controls."""

    if getattr(calcium_plane_cls, _CORE_SCALAR_VALIDATION_INSTALLED_ATTR, False):
        return

    original_position_covariances = calcium_plane_cls.position_covariances
    original_to_constant_velocity_state_moments = calcium_plane_cls.to_constant_velocity_state_moments
    original_build_pairwise_cost_matrix = calcium_plane_cls.build_pairwise_cost_matrix

    @wraps(original_position_covariances)
    def position_covariances(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        args, kwargs = _validated_positional_or_keyword_scalar(
            args,
            kwargs,
            name="regularization",
            positional_index=2,
            default=1.0e-6,
            strictly_positive=False,
        )
        return original_position_covariances(self, *args, **kwargs)

    @wraps(original_to_constant_velocity_state_moments)
    def to_constant_velocity_state_moments(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        args, kwargs = _validated_positional_or_keyword_scalar(
            args,
            kwargs,
            name="velocity_variance",
            positional_index=2,
            default=25.0,
            strictly_positive=False,
        )
        args, kwargs = _validated_positional_or_keyword_scalar(
            args,
            kwargs,
            name="regularization",
            positional_index=3,
            default=1.0e-6,
            strictly_positive=False,
        )
        return original_to_constant_velocity_state_moments(self, *args, **kwargs)

    @wraps(original_build_pairwise_cost_matrix)
    def build_pairwise_cost_matrix(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        kwargs = _validate_pairwise_cost_kwargs(kwargs)
        return original_build_pairwise_cost_matrix(self, *args, **kwargs)

    _mark_wrapper(position_covariances, original_position_covariances)
    _mark_wrapper(to_constant_velocity_state_moments, original_to_constant_velocity_state_moments)
    _mark_wrapper(build_pairwise_cost_matrix, original_build_pairwise_cost_matrix)

    calcium_plane_cls.position_covariances = position_covariances
    calcium_plane_cls.to_constant_velocity_state_moments = to_constant_velocity_state_moments
    calcium_plane_cls.build_pairwise_cost_matrix = build_pairwise_cost_matrix
    setattr(calcium_plane_cls, _CORE_SCALAR_VALIDATION_INSTALLED_ATTR, True)


def _mark_wrapper(wrapper: Callable[..., Any], original: Callable[..., Any]) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)


def _validated_positional_or_keyword_scalar(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    positional_index: int,
    default: float,
    strictly_positive: bool,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if len(args) > positional_index and name in kwargs:
        raise TypeError(f"{name} was specified both positionally and by keyword")

    mutable_args = list(args)
    mutable_kwargs = dict(kwargs)
    if len(mutable_args) > positional_index:
        raw_value = mutable_args[positional_index]
        mutable_args[positional_index] = _validate_finite_scalar(
            name,
            raw_value,
            strictly_positive=strictly_positive,
        )
    else:
        raw_value = mutable_kwargs.get(name, default)
        mutable_kwargs[name] = _validate_finite_scalar(
            name,
            raw_value,
            strictly_positive=strictly_positive,
        )
    return tuple(mutable_args), mutable_kwargs


def _validate_pairwise_cost_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    validated_kwargs = dict(kwargs)
    for name in _NONNEGATIVE_PAIRWISE_CONTROLS:
        if name in validated_kwargs:
            validated_kwargs[name] = _validate_finite_scalar(
                name,
                validated_kwargs[name],
                strictly_positive=False,
            )
    for name in _STRICTLY_POSITIVE_PAIRWISE_CONTROLS:
        if name in validated_kwargs:
            validated_kwargs[name] = _validate_finite_scalar(
                name,
                validated_kwargs[name],
                strictly_positive=True,
            )
    for name in _OPTIONAL_STRICTLY_POSITIVE_PAIRWISE_CONTROLS:
        if name in validated_kwargs and validated_kwargs[name] is not None:
            validated_kwargs[name] = _validate_finite_scalar(
                name,
                validated_kwargs[name],
                strictly_positive=True,
            )
    return validated_kwargs


def _validate_finite_scalar(
    name: str,
    raw_value: Any,
    *,
    strictly_positive: bool,
) -> float:
    requirement = "a finite strictly positive value" if strictly_positive else "a finite non-negative value"
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(f"{name} must be {requirement}")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {requirement}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be {requirement}")
    if strictly_positive:
        if value <= 0.0:
            raise ValueError(f"{name} must be {requirement}")
    elif value < 0.0:
        raise ValueError(f"{name} must be {requirement}")
    return value


__all__ = ["install_core_scalar_validation_patches"]
