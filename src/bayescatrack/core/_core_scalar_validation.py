"""Runtime validation for core geometry and pairwise-cost controls."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

_CORE_SCALAR_VALIDATION_INSTALLED_ATTR = (
    "_bayescatrack_core_scalar_validation_installed"
)
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
_PATCHED_METHOD_NAMES = (
    "roi_areas",
    "centroids",
    "pairwise_centroid_distances",
    "position_covariances",
    "to_measurement_matrix",
    "to_constant_velocity_state_moments",
    "to_export_dict",
    "build_pairwise_cost_matrix",
)


def install_core_scalar_validation_patches(calcium_plane_cls: type[Any]) -> None:
    """Install idempotent validation wrappers for core scalar and boolean controls."""

    installed = getattr(
        calcium_plane_cls, _CORE_SCALAR_VALIDATION_INSTALLED_ATTR, False
    )
    current_methods_are_patched = all(
        _function_chain_has_patch(getattr(calcium_plane_cls, method_name))
        for method_name in _PATCHED_METHOD_NAMES
    )
    if installed and current_methods_are_patched:
        return

    original_roi_areas = calcium_plane_cls.roi_areas
    original_centroids = calcium_plane_cls.centroids
    original_pairwise_centroid_distances = calcium_plane_cls.pairwise_centroid_distances
    original_position_covariances = calcium_plane_cls.position_covariances
    original_to_measurement_matrix = calcium_plane_cls.to_measurement_matrix
    original_to_constant_velocity_state_moments = (
        calcium_plane_cls.to_constant_velocity_state_moments
    )
    original_to_export_dict = calcium_plane_cls.to_export_dict
    original_build_pairwise_cost_matrix = calcium_plane_cls.build_pairwise_cost_matrix

    @wraps(original_roi_areas)
    def roi_areas(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        args, kwargs = _validated_positional_or_keyword_bool(
            args,
            kwargs,
            name="weighted",
            positional_index=0,
        )
        return original_roi_areas(self, *args, **kwargs)

    @wraps(original_centroids)
    def centroids(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        args, kwargs = _validated_positional_or_keyword_bool(
            args,
            kwargs,
            name="weighted",
            positional_index=1,
        )
        return original_centroids(self, *args, **kwargs)

    @wraps(original_pairwise_centroid_distances)
    def pairwise_centroid_distances(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        kwargs = _validated_keyword_bool(kwargs, name="weighted")
        return original_pairwise_centroid_distances(self, *args, **kwargs)

    @wraps(original_position_covariances)
    def position_covariances(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        args, kwargs = _validated_positional_or_keyword_bool(
            args,
            kwargs,
            name="weighted",
            positional_index=1,
        )
        args, kwargs = _validated_positional_or_keyword_scalar(
            args,
            kwargs,
            name="regularization",
            positional_index=2,
            default=1.0e-6,
            strictly_positive=False,
        )
        return original_position_covariances(self, *args, **kwargs)

    @wraps(original_to_measurement_matrix)
    def to_measurement_matrix(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        args, kwargs = _validated_positional_or_keyword_bool(
            args,
            kwargs,
            name="weighted",
            positional_index=1,
        )
        return original_to_measurement_matrix(self, *args, **kwargs)

    @wraps(original_to_constant_velocity_state_moments)
    def to_constant_velocity_state_moments(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        args, kwargs = _validated_positional_or_keyword_bool(
            args,
            kwargs,
            name="weighted",
            positional_index=1,
        )
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

    @wraps(original_to_export_dict)
    def to_export_dict(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        kwargs = _validated_keyword_bool(kwargs, name="weighted")
        kwargs = _validated_keyword_bool(kwargs, name="include_masks")
        return original_to_export_dict(self, *args, **kwargs)

    @wraps(original_build_pairwise_cost_matrix)
    def build_pairwise_cost_matrix(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        kwargs = _validate_pairwise_cost_kwargs(kwargs)
        return original_build_pairwise_cost_matrix(self, *args, **kwargs)

    _mark_wrapper(roi_areas, original_roi_areas)
    _mark_wrapper(centroids, original_centroids)
    _mark_wrapper(pairwise_centroid_distances, original_pairwise_centroid_distances)
    _mark_wrapper(position_covariances, original_position_covariances)
    _mark_wrapper(to_measurement_matrix, original_to_measurement_matrix)
    _mark_wrapper(
        to_constant_velocity_state_moments, original_to_constant_velocity_state_moments
    )
    _mark_wrapper(to_export_dict, original_to_export_dict)
    _mark_wrapper(build_pairwise_cost_matrix, original_build_pairwise_cost_matrix)

    calcium_plane_cls.roi_areas = roi_areas
    calcium_plane_cls.centroids = centroids
    calcium_plane_cls.pairwise_centroid_distances = pairwise_centroid_distances
    calcium_plane_cls.position_covariances = position_covariances
    calcium_plane_cls.to_measurement_matrix = to_measurement_matrix
    calcium_plane_cls.to_constant_velocity_state_moments = (
        to_constant_velocity_state_moments
    )
    calcium_plane_cls.to_export_dict = to_export_dict
    calcium_plane_cls.build_pairwise_cost_matrix = build_pairwise_cost_matrix
    setattr(calcium_plane_cls, _CORE_SCALAR_VALIDATION_INSTALLED_ATTR, True)


def _mark_wrapper(wrapper: Callable[..., Any], original: Callable[..., Any]) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)


def _function_chain_has_patch(function: Any) -> bool:
    seen: set[int] = set()
    current: Any = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, _PATCH_MARKER, False):
            return True
        seen.add(current_id)
        current = getattr(current, _ORIGINAL_ATTR, None)
    return False


def _validated_positional_or_keyword_bool(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    positional_index: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if len(args) > positional_index and name in kwargs:
        raise TypeError(f"{name} was specified both positionally and by keyword")

    mutable_args = list(args)
    mutable_kwargs = dict(kwargs)
    if len(mutable_args) > positional_index:
        mutable_args[positional_index] = _validate_bool_control(
            name, mutable_args[positional_index]
        )
    elif name in mutable_kwargs:
        mutable_kwargs[name] = _validate_bool_control(name, mutable_kwargs[name])
    return tuple(mutable_args), mutable_kwargs


def _validated_keyword_bool(kwargs: dict[str, Any], *, name: str) -> dict[str, Any]:
    if name not in kwargs:
        return kwargs
    validated_kwargs = dict(kwargs)
    validated_kwargs[name] = _validate_bool_control(name, validated_kwargs[name])
    return validated_kwargs


def _validate_bool_control(name: str, raw_value: Any) -> bool:
    if isinstance(raw_value, (bool, np.bool_)):
        return bool(raw_value)
    raise ValueError(f"{name} must be a boolean")


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
    requirement = (
        "a finite positive value"
        if strictly_positive
        else "a finite non-negative value"
    )
    if isinstance(raw_value, np.ndarray):
        if raw_value.shape != ():
            raise ValueError(f"{name} must be {requirement}")
        raw_value = raw_value.item()
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(f"{name} must be {requirement}")
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
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
