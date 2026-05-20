from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import CalciumPlaneData
from .fov_registration import estimate_integer_fov_shift


@dataclass(frozen=True)
class FovAffineEstimate:
    matrix_xy: np.ndarray
    inverse_matrix_xy: np.ndarray
    tile_reference_xy: np.ndarray
    tile_measurement_xy: np.ndarray
    tile_shift_yx: np.ndarray
    tile_peak_correlation: np.ndarray
    tile_residual_norm: np.ndarray
    fit_rmse: float
    fallback_translation: bool = False


@dataclass(frozen=True)
class FovAffineRegistration:
    reference_plane: CalciumPlaneData
    measurement_plane: CalciumPlaneData
    registered_measurement_plane: CalciumPlaneData
    estimate: FovAffineEstimate


def register_measurement_plane_by_fov_affine(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    subtract_mean: bool = True,
    grid_shape: tuple[int, int] = (3, 3),
    min_tile_size: int = 32,
    max_shift_fraction: float = 0.55,
    mask_warp_mode: Literal["nearest", "bilinear"] = "nearest",
) -> FovAffineRegistration:
    if reference_plane.fov is None or measurement_plane.fov is None:
        raise ValueError(
            "Both planes must provide fov images for FOV-affine registration"
        )
    estimate = estimate_fov_affine_transform(
        reference_plane.fov,
        measurement_plane.fov,
        subtract_mean=subtract_mean,
        grid_shape=grid_shape,
        min_tile_size=min_tile_size,
        max_shift_fraction=max_shift_fraction,
    )
    masks = apply_affine_roi_mask_warp(
        measurement_plane.roi_masks,
        estimate.matrix_xy,
        output_shape=reference_plane.image_shape,
        mode=mask_warp_mode,
    )
    registered_fov = apply_affine_image_warp(
        measurement_plane.fov,
        estimate.matrix_xy,
        output_shape=reference_plane.image_shape,
        fill_value=0.0,
    )
    registered_fov[np.abs(registered_fov) < 1.0e-12] = 0.0
    rounded_registered_fov = np.rint(registered_fov)
    registered_fov = np.where(
        np.abs(registered_fov - rounded_registered_fov) < 1.0e-12,
        rounded_registered_fov,
        registered_fov,
    )
    ops = {} if measurement_plane.ops is None else dict(measurement_plane.ops)
    ops.update(
        {
            "registration_backend": "fov-affine",
            "registration_transform_type": "fov-affine",
            "registration_backend_reason": "tile-wise FOV phase-correlation affine fallback",
            "fov_registration_method": "tile_phase_correlation_affine",
            "fov_affine_matrix_xy": estimate.matrix_xy,
            "fov_affine_tile_count": int(estimate.tile_reference_xy.shape[0]),
            "fov_affine_fit_rmse": float(estimate.fit_rmse),
            "fov_affine_fallback_translation": bool(estimate.fallback_translation),
        }
    )
    registered_plane = measurement_plane.with_replaced_masks(
        masks,
        fov=registered_fov,
        source=f"{measurement_plane.source}_fov_affine_registered",
        ops=ops,
    )
    return FovAffineRegistration(
        reference_plane, measurement_plane, registered_plane, estimate
    )


def estimate_fov_affine_transform(
    reference_fov: np.ndarray,
    measurement_fov: np.ndarray,
    *,
    subtract_mean: bool = True,
    grid_shape: tuple[int, int] = (3, 3),
    min_tile_size: int = 32,
    max_shift_fraction: float = 0.55,
) -> FovAffineEstimate:
    reference = np.asarray(reference_fov, dtype=float)
    measurement = np.asarray(measurement_fov, dtype=float)
    if reference.ndim != 2 or measurement.ndim != 2:
        raise ValueError("reference_fov and measurement_fov must be 2-D")
    shape = (
        max(reference.shape[0], measurement.shape[0]),
        max(reference.shape[1], measurement.shape[1]),
    )
    reference = _pad(reference, shape)
    measurement = _pad(measurement, shape)
    ref_xy, meas_xy, shifts_yx, peaks = _tile_correspondences(
        reference,
        measurement,
        subtract_mean=subtract_mean,
        grid_shape=grid_shape,
        min_tile_size=min_tile_size,
        max_shift_fraction=max_shift_fraction,
    )
    if ref_xy.shape[0] < 3 or np.linalg.matrix_rank(_design(meas_xy)) < 3:
        return _translation_estimate(
            reference, measurement, subtract_mean=subtract_mean
        )
    translation_estimate = _translation_estimate(
        reference, measurement, subtract_mean=subtract_mean
    )
    if _tile_shifts_support_global_translation(
        shifts_yx,
        translation_estimate.tile_shift_yx[0],
    ):
        return translation_estimate
    coef, _, _, _ = np.linalg.lstsq(_design(meas_xy), ref_xy, rcond=None)
    matrix_xy = np.asarray(coef.T, dtype=float)
    residual = _design(meas_xy) @ coef - ref_xy
    estimate = _make_estimate(
        matrix_xy, ref_xy, meas_xy, shifts_yx, peaks, residual, False
    )
    if ref_xy.shape[0] > 3:
        keep = _inlier_mask(np.linalg.norm(residual, axis=1))
        if (
            np.count_nonzero(keep) >= 3
            and np.linalg.matrix_rank(_design(meas_xy[keep])) >= 3
        ):
            coef, _, _, _ = np.linalg.lstsq(
                _design(meas_xy[keep]), ref_xy[keep], rcond=None
            )
            matrix_xy = np.asarray(coef.T, dtype=float)
            residual = _design(meas_xy[keep]) @ coef - ref_xy[keep]
            estimate = _make_estimate(
                matrix_xy,
                ref_xy[keep],
                meas_xy[keep],
                shifts_yx[keep],
                peaks[keep],
                residual,
                False,
            )
    return estimate


def apply_affine_image_warp(
    image: np.ndarray,
    matrix_xy: np.ndarray,
    *,
    output_shape: tuple[int, int],
    fill_value: float | bool = 0.0,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Warp a 2-D image into the reference frame by inverse resampling.

    ``matrix_xy`` maps measurement/source ``(x, y)`` coordinates to
    reference/destination coordinates. Sampling the destination grid through the
    inverse transform avoids holes from forward splatting under rotations,
    shears, and scale changes.
    """

    image = np.asarray(image)
    matrix_xy = np.asarray(matrix_xy, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must have shape (height, width)")
    if matrix_xy.shape != (2, 3):
        raise ValueError("matrix_xy must have shape (2, 3)")
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be either 'nearest' or 'bilinear'")

    output_shape = (int(output_shape[0]), int(output_shape[1]))
    source_y, source_x, valid = _affine_output_sample_coordinates(
        matrix_xy,
        source_shape=image.shape,
        output_shape=output_shape,
    )
    if interpolation == "nearest":
        return _nearest_sample_image(
            image,
            source_y,
            source_x,
            valid,
            fill_value=fill_value,
        )
    return _bilinear_sample_image(
        image,
        source_y,
        source_x,
        valid,
        fill_value=float(fill_value),
    )


def apply_affine_roi_mask_warp(
    roi_masks: np.ndarray,
    matrix_xy: np.ndarray,
    *,
    output_shape: tuple[int, int],
    mode: Literal["nearest", "bilinear"] = "nearest",
) -> np.ndarray:
    roi_masks = np.asarray(roi_masks)
    matrix_xy = np.asarray(matrix_xy, dtype=float)
    if roi_masks.ndim != 3:
        raise ValueError("roi_masks must have shape (n_roi, height, width)")
    if matrix_xy.shape != (2, 3):
        raise ValueError("matrix_xy must have shape (2, 3)")
    if mode not in {"nearest", "bilinear"}:
        raise ValueError("mode must be either 'nearest' or 'bilinear'")
    if mode == "bilinear":
        return _apply_affine_roi_mask_warp_bilinear(
            roi_masks,
            matrix_xy,
            output_shape=output_shape,
        )

    source_y, source_x, valid = _affine_output_sample_coordinates(
        matrix_xy,
        source_shape=(int(roi_masks.shape[1]), int(roi_masks.shape[2])),
        output_shape=output_shape,
    )
    output = np.zeros(
        (roi_masks.shape[0], int(output_shape[0]), int(output_shape[1])),
        dtype=roi_masks.dtype,
    )
    if roi_masks.dtype == np.bool_:
        nearest_y = np.zeros(source_y.shape, dtype=int)
        nearest_x = np.zeros(source_x.shape, dtype=int)
        nearest_y[valid] = np.clip(
            np.rint(source_y[valid]).astype(int), 0, roi_masks.shape[1] - 1
        )
        nearest_x[valid] = np.clip(
            np.rint(source_x[valid]).astype(int), 0, roi_masks.shape[2] - 1
        )
        for roi_index, mask in enumerate(roi_masks):
            sampled = np.zeros(output_shape, dtype=bool)
            sampled[valid] = mask[nearest_y[valid], nearest_x[valid]] > 0
            output[roi_index] = sampled
        return output

    for roi_index, mask in enumerate(roi_masks):
        sampled = _bilinear_sample_image(
            np.asarray(mask, dtype=float),
            source_y,
            source_x,
            valid,
            fill_value=0.0,
        )
        output[roi_index] = sampled.astype(output.dtype, copy=False)
    return output


def _apply_affine_roi_mask_warp_bilinear(
    roi_masks: np.ndarray,
    matrix_xy: np.ndarray,
    *,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Splat ROI pixels with bilinear weights to preserve soft mask evidence."""

    roi_masks = np.asarray(roi_masks)
    matrix_xy = np.asarray(matrix_xy, dtype=float)
    output = np.zeros(
        (roi_masks.shape[0], int(output_shape[0]), int(output_shape[1])),
        dtype=float,
    )
    linear = matrix_xy[:, :2]
    offset = matrix_xy[:, 2]
    for roi_index, mask in enumerate(roi_masks):
        yy, xx = np.nonzero(mask)
        if yy.size == 0:
            continue
        src_xy = np.column_stack((xx.astype(float), yy.astype(float)))
        dst_xy = src_xy @ linear.T + offset[None, :]
        values = np.asarray(mask[yy, xx], dtype=float)
        _bilinear_splat(output[roi_index], dst_xy[:, 0], dst_xy[:, 1], values)
    return output


def _bilinear_splat(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, values: np.ndarray
) -> None:
    height, width = image.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    for dx in (0, 1):
        for dy in (0, 1):
            xi = x0 + dx
            yi = y0 + dy
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            if not np.any(valid):
                continue
            wx = 1.0 - np.abs(x[valid] - xi[valid])
            wy = 1.0 - np.abs(y[valid] - yi[valid])
            weights = np.clip(wx, 0.0, 1.0) * np.clip(wy, 0.0, 1.0)
            np.add.at(image, (yi[valid], xi[valid]), values[valid] * weights)


def _affine_output_sample_coordinates(
    matrix_xy: np.ndarray,
    *,
    source_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse_xy = invert_affine_xy(matrix_xy)
    yy, xx = np.indices(
        (int(output_shape[0]), int(output_shape[1])),
        dtype=float,
    )
    query_xy = np.column_stack((xx.ravel(), yy.ravel()))
    source_xy = (
        query_xy @ np.asarray(inverse_xy[:, :2], dtype=float).T
        + np.asarray(inverse_xy[:, 2], dtype=float)[None, :]
    )
    source_x = source_xy[:, 0].reshape(output_shape)
    source_y = source_xy[:, 1].reshape(output_shape)
    valid = (
        np.isfinite(source_y)
        & np.isfinite(source_x)
        & (source_y >= 0.0)
        & (source_x >= 0.0)
        & (source_y <= max(int(source_shape[0]) - 1, 0))
        & (source_x <= max(int(source_shape[1]) - 1, 0))
    )
    return source_y, source_x, valid


def _nearest_sample_image(
    image: np.ndarray,
    source_y: np.ndarray,
    source_x: np.ndarray,
    valid: np.ndarray,
    *,
    fill_value: float | bool,
) -> np.ndarray:
    output = np.full(source_y.shape, fill_value, dtype=image.dtype)
    if not np.any(valid):
        return output
    y = np.rint(source_y[valid]).astype(int)
    x = np.rint(source_x[valid]).astype(int)
    y = np.clip(y, 0, image.shape[0] - 1)
    x = np.clip(x, 0, image.shape[1] - 1)
    output[valid] = image[y, x]
    return output


def _bilinear_sample_image(
    image: np.ndarray,
    source_y: np.ndarray,
    source_x: np.ndarray,
    valid: np.ndarray,
    *,
    fill_value: float,
) -> np.ndarray:
    output = np.full(source_y.shape, float(fill_value), dtype=float)
    if not np.any(valid):
        return output

    y = source_y[valid]
    x = source_x[valid]
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.clip(y0 + 1, 0, image.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    x0 = np.clip(x0, 0, image.shape[1] - 1)
    wy = y - y0
    wx = x - x0
    output[valid] = (
        (1.0 - wy) * (1.0 - wx) * image[y0, x0]
        + (1.0 - wy) * wx * image[y0, x1]
        + wy * (1.0 - wx) * image[y1, x0]
        + wy * wx * image[y1, x1]
    )
    return output


def invert_affine_xy(matrix_xy: np.ndarray) -> np.ndarray:
    linear = np.asarray(matrix_xy, dtype=float)[:, :2]
    offset = np.asarray(matrix_xy, dtype=float)[:, 2]
    try:
        inv_linear = np.linalg.inv(linear)
    except np.linalg.LinAlgError:
        inv_linear = np.linalg.pinv(linear)
    return np.column_stack((inv_linear, -inv_linear @ offset))


def _tile_correspondences(
    reference,
    measurement,
    *,
    subtract_mean,
    grid_shape,
    min_tile_size,
    max_shift_fraction,
):
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    y_edges = np.linspace(0, reference.shape[0], rows + 1, dtype=int)
    x_edges = np.linspace(0, reference.shape[1], cols + 1, dtype=int)
    ref_points, meas_points, shifts, peaks = [], [], [], []
    for row in range(rows):
        for col in range(cols):
            ys = slice(int(y_edges[row]), int(y_edges[row + 1]))
            xs = slice(int(x_edges[col]), int(x_edges[col + 1]))
            if ys.stop - ys.start < min_tile_size or xs.stop - xs.start < min_tile_size:
                continue
            ref_tile = reference[ys, xs]
            meas_tile = measurement[ys, xs]
            if _low_info(ref_tile) or _low_info(meas_tile):
                continue
            shift_yx, peak = estimate_integer_fov_shift(
                ref_tile, meas_tile, subtract_mean=subtract_mean
            )
            max_shift = max(abs(int(shift_yx[0])), abs(int(shift_yx[1])))
            if (
                max_shift
                > max(ys.stop - ys.start, xs.stop - xs.start) * max_shift_fraction
            ):
                continue
            center = np.asarray(
                [0.5 * (xs.start + xs.stop - 1), 0.5 * (ys.start + ys.stop - 1)],
                dtype=float,
            )
            shift_xy = np.asarray([float(shift_yx[1]), float(shift_yx[0])], dtype=float)
            ref_points.append(center)
            meas_points.append(center - shift_xy)
            shifts.append(shift_yx.astype(float))
            peaks.append(float(peak))
    if not ref_points:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))
    return (
        np.vstack(ref_points),
        np.vstack(meas_points),
        np.vstack(shifts),
        np.asarray(peaks, dtype=float),
    )


def _translation_estimate(reference, measurement, *, subtract_mean):
    shift_yx, peak = estimate_integer_fov_shift(
        reference, measurement, subtract_mean=subtract_mean
    )
    matrix_xy = np.asarray(
        [[1.0, 0.0, float(shift_yx[1])], [0.0, 1.0, float(shift_yx[0])]], dtype=float
    )
    return _make_estimate(
        matrix_xy,
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        np.asarray(shift_yx, dtype=float).reshape(1, 2),
        np.asarray([peak]),
        np.zeros((0, 2)),
        True,
    )


def _tile_shifts_support_global_translation(
    shifts_yx: np.ndarray, global_shift_yx: np.ndarray
) -> bool:
    shifts_yx = np.asarray(shifts_yx, dtype=float)
    if shifts_yx.size == 0:
        return False
    global_shift_yx = np.rint(np.asarray(global_shift_yx, dtype=float)).astype(int)
    rounded_shifts = np.rint(shifts_yx).astype(int)
    matches = np.all(rounded_shifts == global_shift_yx[None, :], axis=1)
    min_matches = max(3, int(np.ceil(0.75 * rounded_shifts.shape[0])))
    return int(np.count_nonzero(matches)) >= min_matches


def _make_estimate(matrix_xy, ref_xy, meas_xy, shifts_yx, peaks, residual, fallback):
    matrix_xy = _snap_near_integer_affine(matrix_xy)
    residual_norm = (
        np.linalg.norm(residual, axis=1)
        if residual.size
        else np.zeros((0,), dtype=float)
    )
    return FovAffineEstimate(
        matrix_xy,
        invert_affine_xy(matrix_xy),
        np.asarray(ref_xy, dtype=float),
        np.asarray(meas_xy, dtype=float),
        np.asarray(shifts_yx, dtype=float),
        np.asarray(peaks, dtype=float),
        residual_norm,
        float(np.sqrt(np.mean(residual_norm**2))) if residual_norm.size else 0.0,
        fallback,
    )


def _snap_near_integer_affine(matrix_xy: np.ndarray) -> np.ndarray:
    matrix_xy = np.asarray(matrix_xy, dtype=float).copy()
    nearest = np.rint(matrix_xy)
    close = np.abs(matrix_xy - nearest) < 1.0e-8
    matrix_xy[close] = nearest[close]
    return matrix_xy


def _design(xy):
    return np.column_stack((xy, np.ones((xy.shape[0],), dtype=float)))


def _inlier_mask(residual_norm):
    finite = residual_norm[np.isfinite(residual_norm)]
    if finite.size == 0:
        return np.ones_like(residual_norm, dtype=bool)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    return np.isfinite(residual_norm) & (
        residual_norm <= max(med + 3.0 * 1.4826 * mad, np.percentile(finite, 75), 1.0)
    )


def _low_info(tile):
    finite = np.asarray(tile, dtype=float)
    finite = finite[np.isfinite(finite)]
    return finite.size == 0 or float(np.std(finite)) <= 1.0e-8


def _pad(image, shape):
    if image.shape == shape:
        return image
    out = np.zeros(shape, dtype=image.dtype)
    out[: image.shape[0], : image.shape[1]] = image
    return out
