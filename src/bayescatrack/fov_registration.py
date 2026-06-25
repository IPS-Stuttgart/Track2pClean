from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy import ndimage, optimize

from .core.bridge import (
    CalciumPlaneData,
    SessionAssociationBundle,
    Track2pSession,
    build_consecutive_session_association_bundles,
    build_session_pair_association_bundle,
)


@dataclass(frozen=True)
class FovTranslationRegistration:
    reference_plane: CalciumPlaneData
    measurement_plane: CalciumPlaneData
    registered_measurement_plane: CalciumPlaneData
    measurement_to_reference_shift_yx: np.ndarray
    reference_to_measurement_shift_yx: np.ndarray
    peak_correlation: float


@dataclass(frozen=True)
class FovRegisteredSessionPairBundle:
    registration: FovTranslationRegistration
    association_bundle: SessionAssociationBundle


MaskInterpolation = Literal["nearest", "bilinear"]


@dataclass(frozen=True)
class _FovAssociationOptions:
    subtract_mean: bool = True
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1e-6
    pairwise_cost_kwargs: Mapping[str, Any] | None = None
    return_pairwise_components: bool = True
    subpixel: bool = False
    subpixel_refinement_radius: float = 1.0
    subpixel_interpolation_order: int = 1
    mask_interpolation: MaskInterpolation = "bilinear"

    def registration_kwargs(self) -> dict[str, Any]:
        return {
            "subtract_mean": self.subtract_mean,
            "subpixel": self.subpixel,
            "subpixel_refinement_radius": self.subpixel_refinement_radius,
            "subpixel_interpolation_order": self.subpixel_interpolation_order,
            "mask_interpolation": self.mask_interpolation,
        }

    def association_kwargs(self) -> dict[str, Any]:
        pairwise_cost_kwargs = self.pairwise_cost_kwargs
        if self.subpixel and self.mask_interpolation == "bilinear":
            pairwise_cost_kwargs = dict(pairwise_cost_kwargs or {})
            pairwise_cost_kwargs.setdefault("soft_iou", True)
        return {
            "order": self.order,
            "weighted_centroids": self.weighted_centroids,
            "velocity_variance": self.velocity_variance,
            "regularization": self.regularization,
            "pairwise_cost_kwargs": pairwise_cost_kwargs,
            "return_pairwise_components": self.return_pairwise_components,
        }


_FOV_ASSOCIATION_OPTION_DEFAULTS: dict[str, Any] = {
    "subtract_mean": True,
    "order": "xy",
    "weighted_centroids": False,
    "velocity_variance": 25.0,
    "regularization": 1e-6,
    "pairwise_cost_kwargs": None,
    "return_pairwise_components": True,
    "subpixel": False,
    "subpixel_refinement_radius": 1.0,
    "subpixel_interpolation_order": 1,
    "mask_interpolation": "bilinear",
}


def _fov_association_options_from_kwargs(
    bundle_kwargs: Mapping[str, Any],
) -> _FovAssociationOptions:
    normalized_kwargs = dict(bundle_kwargs)
    if "subpixel_refinement" in normalized_kwargs:
        subpixel_refinement_value = normalized_kwargs.pop("subpixel_refinement")
        if subpixel_refinement_value is not None:
            subpixel_refinement = _strict_bool(
                subpixel_refinement_value, name="subpixel_refinement"
            )
            if (
                "subpixel" in normalized_kwargs
                and _strict_bool(normalized_kwargs["subpixel"], name="subpixel")
                != subpixel_refinement
            ):
                raise ValueError("subpixel and subpixel_refinement disagree")
            normalized_kwargs["subpixel"] = subpixel_refinement
    unexpected_names = sorted(
        set(normalized_kwargs).difference(_FOV_ASSOCIATION_OPTION_DEFAULTS)
    )
    if unexpected_names:
        joined_names = ", ".join(unexpected_names)
        raise TypeError(f"Unexpected FOV registration option(s): {joined_names}")
    options = _FovAssociationOptions(
        **{
            **_FOV_ASSOCIATION_OPTION_DEFAULTS,
            **normalized_kwargs,
        }
    )
    _strict_bool(options.subtract_mean, name="subtract_mean")
    _strict_bool(options.weighted_centroids, name="weighted_centroids")
    _strict_bool(options.return_pairwise_components, name="return_pairwise_components")
    _strict_bool(options.subpixel, name="subpixel")
    _finite_nonnegative_float(
        options.subpixel_refinement_radius, name="subpixel_refinement_radius"
    )
    _validate_subpixel_interpolation_order(options.subpixel_interpolation_order)
    _validate_mask_interpolation(options.mask_interpolation)
    return options


def _prepare_fov_phase_correlation_inputs(
    reference_fov: np.ndarray,
    measurement_fov: np.ndarray,
    *,
    subtract_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    subtract_mean = _strict_bool(subtract_mean, name="subtract_mean")
    reference = np.asarray(reference_fov, dtype=float)
    measurement = np.asarray(measurement_fov, dtype=float)
    if reference.ndim != 2 or measurement.ndim != 2:
        raise ValueError("reference_fov and measurement_fov must both be 2-D arrays")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(measurement)):
        raise ValueError(
            "reference_fov and measurement_fov must contain only finite values"
        )
    if np.ptp(reference) <= 0.0 or np.ptp(measurement) <= 0.0:
        raise ValueError(
            "reference_fov and measurement_fov must contain spatial variation "
            "for phase-correlation registration"
        )
    if reference.shape != measurement.shape:
        common_shape = (
            max(int(reference.shape[0]), int(measurement.shape[0])),
            max(int(reference.shape[1]), int(measurement.shape[1])),
        )
        reference = _pad_image_to_shape(reference, common_shape)
        measurement = _pad_image_to_shape(measurement, common_shape)
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(measurement)):
        raise ValueError(
            "reference_fov and measurement_fov must contain only finite values"
        )
    if subtract_mean:
        reference = reference - float(np.mean(reference))
        measurement = measurement - float(np.mean(measurement))

    if (
        float(np.linalg.norm(reference.ravel())) <= np.finfo(float).eps
        or float(np.linalg.norm(measurement.ravel())) <= np.finfo(float).eps
    ):
        raise ValueError("Cannot estimate FOV shift from constant or empty FOV images")
    return reference, measurement


def _phase_correlation_surface(
    reference: np.ndarray,
    measurement: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], float]:
    cross_power = np.fft.fftn(reference) * np.conj(np.fft.fftn(measurement))
    magnitude = np.abs(cross_power)
    magnitude[magnitude == 0.0] = 1.0
    correlation = np.abs(np.fft.ifftn(cross_power / magnitude))
    peak_index = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    return correlation, peak_index, float(correlation[peak_index])


def _signed_phase_peak_shift(
    peak_index: Sequence[int] | np.ndarray,
    image_shape: Sequence[int] | np.ndarray,
) -> np.ndarray:
    shift_yx = np.asarray(peak_index, dtype=float)
    for axis, size in enumerate(tuple(int(value) for value in image_shape)):
        if shift_yx[axis] > size // 2:
            shift_yx[axis] -= size
    return shift_yx


def estimate_integer_fov_shift(
    reference_fov: np.ndarray,
    measurement_fov: np.ndarray,
    *,
    subtract_mean: bool = True,
) -> tuple[np.ndarray, float]:
    """Return the integer shift to apply to ``measurement_fov`` to align it with ``reference_fov``."""

    reference, measurement = _prepare_fov_phase_correlation_inputs(
        reference_fov,
        measurement_fov,
        subtract_mean=subtract_mean,
    )
    correlation, peak_index, peak_correlation = _phase_correlation_surface(
        reference,
        measurement,
    )
    del correlation
    return (
        _signed_phase_peak_shift(peak_index, reference.shape).astype(int),
        peak_correlation,
    )


def estimate_subpixel_fov_shift(
    reference_fov: np.ndarray,
    measurement_fov: np.ndarray,
    *,
    subtract_mean: bool = True,
    refinement_radius: float = 1.0,
    interpolation_order: int = 1,
) -> tuple[np.ndarray, float]:
    """Return a subpixel shift to apply to ``measurement_fov``."""

    refinement_radius = _finite_nonnegative_float(
        refinement_radius, name="refinement_radius"
    )
    interpolation_order = _validate_subpixel_interpolation_order(interpolation_order)

    reference, measurement = _prepare_fov_phase_correlation_inputs(
        reference_fov,
        measurement_fov,
        subtract_mean=subtract_mean,
    )
    _, peak_index, peak_correlation = _phase_correlation_surface(reference, measurement)
    integer_shift = _signed_phase_peak_shift(peak_index, reference.shape).astype(float)
    if refinement_radius == 0.0:
        return integer_shift, peak_correlation
    return _refine_shift_by_normalized_correlation(
        reference,
        measurement,
        initial_shift_yx=integer_shift,
        refinement_radius=refinement_radius,
        interpolation_order=interpolation_order,
    )


def estimate_fov_shift(
    reference_fov: np.ndarray,
    measurement_fov: np.ndarray,
    *,
    subtract_mean: bool = True,
    subpixel_refinement: bool = True,
    subpixel_refinement_radius: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Return the FOV shift, preserving the older subpixel-refinement API."""

    subpixel_refinement = _strict_bool(subpixel_refinement, name="subpixel_refinement")
    if subpixel_refinement:
        return estimate_subpixel_fov_shift(
            reference_fov,
            measurement_fov,
            subtract_mean=subtract_mean,
            refinement_radius=subpixel_refinement_radius,
        )
    return estimate_integer_fov_shift(
        reference_fov,
        measurement_fov,
        subtract_mean=subtract_mean,
    )


def _refine_shift_by_normalized_correlation(
    reference: np.ndarray,
    measurement: np.ndarray,
    *,
    initial_shift_yx: np.ndarray,
    refinement_radius: float,
    interpolation_order: int,
) -> tuple[np.ndarray, float]:
    initial_shift_yx = np.asarray(initial_shift_yx, dtype=float).reshape(2)
    initial_registered = apply_subpixel_image_translation(
        measurement,
        initial_shift_yx,
        output_shape=reference.shape,
        interpolation_order=interpolation_order,
    )
    initial_score = _normalized_fov_correlation(reference, initial_registered)

    def objective(candidate_shift: np.ndarray) -> float:
        registered = apply_subpixel_image_translation(
            measurement,
            candidate_shift,
            output_shape=reference.shape,
            interpolation_order=interpolation_order,
        )
        score = _normalized_fov_correlation(reference, registered)
        if not np.isfinite(score):
            return 1.0e6
        return -score

    bounds = tuple(
        (float(value - refinement_radius), float(value + refinement_radius))
        for value in initial_shift_yx
    )
    try:
        result = optimize.minimize(
            objective,
            initial_shift_yx,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 60, "xtol": 1.0e-3, "ftol": 1.0e-6},
        )
    except (RuntimeError, ValueError):
        return initial_shift_yx, initial_score

    if np.all(np.isfinite(result.x)) and np.isfinite(result.fun):
        refined_score = -float(result.fun)
        if refined_score >= initial_score:
            return np.asarray(result.x, dtype=float), refined_score
    return initial_shift_yx, initial_score


def _normalized_fov_correlation(
    reference: np.ndarray, measurement: np.ndarray
) -> float:
    reference = np.asarray(reference, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    reference_centered = reference - float(np.mean(reference))
    measurement_centered = measurement - float(np.mean(measurement))
    denominator = float(
        np.linalg.norm(reference_centered.ravel())
        * np.linalg.norm(measurement_centered.ravel())
    )
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(
        np.dot(reference_centered.ravel(), measurement_centered.ravel()) / denominator
    )


def _pad_image_to_shape(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if image.shape == output_shape:
        return image
    result = np.zeros(output_shape, dtype=image.dtype)
    result[: image.shape[0], : image.shape[1]] = image
    return result


def _overlap_slices(
    source_size: int, output_size: int, shift: int
) -> tuple[slice, slice]:
    source_start = max(0, -shift)
    source_stop = min(source_size, output_size - shift)
    length = max(0, source_stop - source_start)
    if length == 0:
        empty_source = min(max(source_start, 0), source_size)
        empty_destination = min(max(shift, 0), output_size)
        return slice(empty_source, empty_source), slice(
            empty_destination, empty_destination
        )
    source_stop = source_start + length
    destination_start = max(0, shift)
    destination_stop = destination_start + length
    return slice(source_start, source_stop), slice(destination_start, destination_stop)


def apply_integer_image_translation(
    image: np.ndarray,
    shift_yx: Sequence[int] | np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    fill_value: float | bool = 0.0,
) -> np.ndarray:
    """Translate one 2-D image with zero padding."""

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must have shape (height, width)")
    shift_y, shift_x = (int(v) for v in np.asarray(shift_yx, dtype=int))
    if output_shape is None:
        output_shape = image.shape
    result = np.full(output_shape, fill_value, dtype=image.dtype)
    src_y, dst_y = _overlap_slices(image.shape[0], output_shape[0], shift_y)
    src_x, dst_x = _overlap_slices(image.shape[1], output_shape[1], shift_x)
    result[dst_y, dst_x] = image[src_y, src_x]
    return result


def _strict_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative value") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return numeric


def _validate_subpixel_interpolation_order(interpolation_order: int) -> int:
    if isinstance(interpolation_order, (bool, np.bool_)):
        raise ValueError(
            "subpixel interpolation order must be an integer between 0 and 5"
        )
    if isinstance(interpolation_order, (float, np.floating)):
        if (
            not np.isfinite(interpolation_order)
            or not float(interpolation_order).is_integer()
        ):
            raise ValueError(
                "subpixel interpolation order must be an integer between 0 and 5"
            )
        order = int(interpolation_order)
    elif isinstance(interpolation_order, str):
        stripped = interpolation_order.strip()
        try:
            numeric_order = float(stripped)
        except ValueError as exc:
            raise ValueError(
                "subpixel interpolation order must be an integer between 0 and 5"
            ) from exc
        if not np.isfinite(numeric_order) or not numeric_order.is_integer():
            raise ValueError(
                "subpixel interpolation order must be an integer between 0 and 5"
            )
        order = int(numeric_order)
    else:
        try:
            order = operator.index(interpolation_order)
        except TypeError as exc:
            raise ValueError(
                "subpixel interpolation order must be an integer between 0 and 5"
            ) from exc
    if order < 0 or order > 5:
        raise ValueError(
            "subpixel interpolation order must be an integer between 0 and 5"
        )
    return order


def _validate_mask_interpolation(interpolation: str) -> None:
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("mask_interpolation must be either 'nearest' or 'bilinear'")


def _interpolation_order_from_mask_interpolation(interpolation: str) -> int:
    _validate_mask_interpolation(interpolation)
    return 0 if interpolation == "nearest" else 1


def apply_subpixel_image_translation(
    image: np.ndarray,
    shift_yx: Sequence[float] | np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    fill_value: float = 0.0,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Translate one 2-D image by a potentially fractional shift with constant padding."""

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must have shape (height, width)")
    if output_shape is None:
        output_shape = image.shape
    if len(output_shape) != 2:
        raise ValueError("output_shape must have length 2")
    shift = np.asarray(shift_yx, dtype=float).reshape(2)
    if not np.all(np.isfinite(shift)):
        raise ValueError("shift_yx must contain finite values")
    interpolation_order = _validate_subpixel_interpolation_order(interpolation_order)
    return np.asarray(
        ndimage.affine_transform(
            np.asarray(image, dtype=float),
            matrix=np.eye(2),
            offset=-shift,
            output_shape=(int(output_shape[0]), int(output_shape[1])),
            order=interpolation_order,
            mode="constant",
            cval=float(fill_value),
            prefilter=interpolation_order > 1,
        ),
        dtype=float,
    )


def apply_image_translation(
    image: np.ndarray,
    shift_yx: Sequence[float] | np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    fill_value: float | bool = 0.0,
    interpolation: MaskInterpolation = "bilinear",
) -> np.ndarray:
    """Translate one 2-D image, accepting the older interpolation names."""

    return apply_subpixel_image_translation(
        image,
        shift_yx,
        output_shape=output_shape,
        fill_value=float(fill_value),
        interpolation_order=_interpolation_order_from_mask_interpolation(interpolation),
    )


def apply_integer_roi_mask_translation(
    roi_masks: np.ndarray,
    shift_yx: Sequence[int] | np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Translate an ROI-mask stack with zero padding."""

    roi_masks = np.asarray(roi_masks)
    if roi_masks.ndim != 3:
        raise ValueError("roi_masks must have shape (n_roi, height, width)")
    if output_shape is None:
        output_shape = (int(roi_masks.shape[1]), int(roi_masks.shape[2]))
    translated = np.zeros((roi_masks.shape[0], *output_shape), dtype=roi_masks.dtype)
    fill_value = False if roi_masks.dtype == np.bool_ else 0.0
    for roi_index, mask in enumerate(roi_masks):
        translated[roi_index] = apply_integer_image_translation(
            mask,
            shift_yx,
            output_shape=output_shape,
            fill_value=fill_value,
        )
    return translated


def apply_subpixel_roi_mask_translation(
    roi_masks: np.ndarray,
    shift_yx: Sequence[float] | np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Translate an ROI-mask stack by a potentially fractional shift."""

    roi_masks = np.asarray(roi_masks)
    if roi_masks.ndim != 3:
        raise ValueError("roi_masks must have shape (n_roi, height, width)")
    if output_shape is None:
        output_shape = (int(roi_masks.shape[1]), int(roi_masks.shape[2]))
    translated = np.zeros((roi_masks.shape[0], *output_shape), dtype=float)
    for roi_index, mask in enumerate(roi_masks):
        translated[roi_index] = apply_subpixel_image_translation(
            mask,
            shift_yx,
            output_shape=output_shape,
            fill_value=0.0,
            interpolation_order=interpolation_order,
        )
    return translated


def apply_roi_mask_translation(
    roi_masks: np.ndarray,
    shift_yx: Sequence[float] | np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    interpolation: MaskInterpolation = "bilinear",
) -> np.ndarray:
    """Translate an ROI-mask stack, accepting the older interpolation names."""

    return apply_subpixel_roi_mask_translation(
        roi_masks,
        shift_yx,
        output_shape=output_shape,
        interpolation_order=_interpolation_order_from_mask_interpolation(interpolation),
    )


def register_measurement_plane_by_fov_translation(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    subtract_mean: bool = True,
    subpixel: bool = False,
    subpixel_refinement: bool | None = None,
    subpixel_refinement_radius: float = 1.0,
    subpixel_interpolation_order: int = 1,
    mask_interpolation: MaskInterpolation = "bilinear",
) -> FovTranslationRegistration:
    """Align ``measurement_plane`` to ``reference_plane`` using FOV phase correlation."""

    if reference_plane.fov is None or measurement_plane.fov is None:
        raise ValueError(
            "Both planes must provide fov images for FOV-based registration"
        )
    subpixel = _strict_bool(subpixel, name="subpixel")
    if subpixel_refinement is not None:
        subpixel = _strict_bool(subpixel_refinement, name="subpixel_refinement")
    subpixel_refinement_radius = _finite_nonnegative_float(
        subpixel_refinement_radius, name="subpixel_refinement_radius"
    )
    subpixel_interpolation_order = _validate_subpixel_interpolation_order(
        subpixel_interpolation_order
    )
    mask_interpolation_order = _interpolation_order_from_mask_interpolation(
        mask_interpolation
    )
    try:
        if subpixel:
            shift_yx, peak_correlation = estimate_subpixel_fov_shift(
                reference_plane.fov,
                measurement_plane.fov,
                subtract_mean=subtract_mean,
                refinement_radius=subpixel_refinement_radius,
                interpolation_order=subpixel_interpolation_order,
            )
        else:
            shift_yx, peak_correlation = estimate_integer_fov_shift(
                reference_plane.fov,
                measurement_plane.fov,
                subtract_mean=subtract_mean,
            )
    except ValueError as exc:
        if not _is_constant_fov_registration_error(exc):
            raise
        shift_yx = np.zeros(2, dtype=float if subpixel else int)
        peak_correlation = 0.0
    shift_dtype = float if subpixel else int
    shift_yx = np.asarray(shift_yx, dtype=shift_dtype)
    if subpixel:
        registered_masks = apply_subpixel_roi_mask_translation(
            measurement_plane.roi_masks,
            shift_yx,
            output_shape=reference_plane.image_shape,
            interpolation_order=mask_interpolation_order,
        )
        registered_fov = apply_subpixel_image_translation(
            measurement_plane.fov,
            shift_yx,
            output_shape=reference_plane.image_shape,
            fill_value=0.0,
            interpolation_order=subpixel_interpolation_order,
        )
        registration_method = "phase_correlation_subpixel_translation"
    else:
        registered_masks = apply_integer_roi_mask_translation(
            measurement_plane.roi_masks,
            shift_yx,
            output_shape=reference_plane.image_shape,
        )
        registered_fov = apply_integer_image_translation(
            measurement_plane.fov,
            shift_yx,
            output_shape=reference_plane.image_shape,
            fill_value=0.0,
        )
        registration_method = "phase_correlation_translation"
    registration_ops = (
        {} if measurement_plane.ops is None else dict(measurement_plane.ops)
    )
    registration_ops.update(
        {
            "fov_registration_method": registration_method,
            "fov_registration_measurement_to_reference_shift_yx": shift_yx.astype(
                shift_dtype
            ),
            "fov_registration_reference_to_measurement_shift_yx": (-shift_yx).astype(
                shift_dtype
            ),
            "fov_registration_peak_correlation": peak_correlation,
        }
    )
    if subpixel:
        registration_ops.update(
            {
                "fov_registration_subpixel_refinement_radius": float(
                    subpixel_refinement_radius
                ),
                "fov_registration_subpixel_interpolation_order": int(
                    subpixel_interpolation_order
                ),
                "fov_registration_subpixel_refinement": True,
                "fov_registration_mask_interpolation": mask_interpolation,
            }
        )
    registered_plane = measurement_plane.with_replaced_masks(
        registered_masks,
        fov=registered_fov,
        source=f"{measurement_plane.source}_fov_registered",
        ops=registration_ops,
    )
    return FovTranslationRegistration(
        reference_plane=reference_plane,
        measurement_plane=measurement_plane,
        registered_measurement_plane=registered_plane,
        measurement_to_reference_shift_yx=shift_yx.astype(shift_dtype),
        reference_to_measurement_shift_yx=(-shift_yx).astype(shift_dtype),
        peak_correlation=peak_correlation,
    )


def _is_constant_fov_registration_error(exc: ValueError) -> bool:
    message = str(exc)
    return (
        "constant or empty FOV images" in message
        or "spatial variation for phase-correlation registration" in message
    )


# pylint: disable=too-many-arguments
def build_fov_registered_session_pair_association_bundle(
    reference_session: Track2pSession,
    measurement_session: Track2pSession,
    *,
    subtract_mean: bool = True,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = True,
    subpixel: bool = False,
    subpixel_refinement: bool | None = None,
    subpixel_refinement_radius: float = 1.0,
    subpixel_interpolation_order: int = 1,
    mask_interpolation: MaskInterpolation = "bilinear",
) -> FovRegisteredSessionPairBundle:
    """Register the later session by FOV, then build the standard association bundle."""

    options = _fov_association_options_from_kwargs(
        {
            "subtract_mean": subtract_mean,
            "order": order,
            "weighted_centroids": weighted_centroids,
            "velocity_variance": velocity_variance,
            "regularization": regularization,
            "pairwise_cost_kwargs": pairwise_cost_kwargs,
            "return_pairwise_components": return_pairwise_components,
            "subpixel": subpixel,
            "subpixel_refinement": subpixel_refinement,
            "subpixel_refinement_radius": subpixel_refinement_radius,
            "subpixel_interpolation_order": subpixel_interpolation_order,
            "mask_interpolation": mask_interpolation,
        }
    )
    registration = register_measurement_plane_by_fov_translation(
        reference_session.plane_data,
        measurement_session.plane_data,
        **options.registration_kwargs(),
    )
    association_bundle = build_session_pair_association_bundle(
        reference_session,
        measurement_session,
        measurement_plane_in_reference_frame=registration.registered_measurement_plane,
        **options.association_kwargs(),
    )
    return FovRegisteredSessionPairBundle(
        registration=registration,
        association_bundle=association_bundle,
    )


def build_fov_registered_consecutive_session_association_bundles(
    sessions: Sequence[Track2pSession],
    **bundle_kwargs: Any,
) -> list[FovRegisteredSessionPairBundle]:
    """Build FOV-registered association bundles for all consecutive session pairs."""

    options = _fov_association_options_from_kwargs(bundle_kwargs)
    session_list = list(sessions)
    registrations = [
        register_measurement_plane_by_fov_translation(
            session_list[pair_index].plane_data,
            session_list[pair_index + 1].plane_data,
            **options.registration_kwargs(),
        )
        for pair_index in range(len(session_list) - 1)
    ]
    association_bundles = build_consecutive_session_association_bundles(
        session_list,
        measurement_planes_in_reference_frames=[
            registration.registered_measurement_plane for registration in registrations
        ],
        **options.association_kwargs(),
    )
    return [
        FovRegisteredSessionPairBundle(
            registration=registration,
            association_bundle=association_bundle,
        )
        for registration, association_bundle in zip(registrations, association_bundles)
    ]
