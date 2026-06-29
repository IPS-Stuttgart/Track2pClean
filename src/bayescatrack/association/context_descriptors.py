"""Local FOV and graph-context descriptors for ROI association."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContextDescriptorConfig:
    """Parameters for lightweight context descriptors."""

    patch_radius: int = 10
    neighbor_k: int = 8
    density_radius: float = 20.0
    histogram_bins: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "patch_radius",
            _nonnegative_int(self.patch_radius, name="patch_radius"),
        )
        object.__setattr__(
            self,
            "neighbor_k",
            _positive_int(self.neighbor_k, name="neighbor_k"),
        )
        object.__setattr__(
            self,
            "density_radius",
            _positive_float(self.density_radius, name="density_radius"),
        )
        object.__setattr__(
            self,
            "histogram_bins",
            _integer_at_least(
                self.histogram_bins,
                minimum=2,
                name="histogram_bins",
                message="histogram_bins must be greater than one",
            ),
        )


def roi_context_descriptors(
    plane: Any,
    *,
    order: str = "xy",
    weighted_centroids: bool = False,
    config: ContextDescriptorConfig | None = None,
) -> np.ndarray:
    """Return a descriptor matrix with FOV patch, density, and graph features."""

    cfg = config or ContextDescriptorConfig()
    centroids = _centroids_xy(plane, order=order, weighted=weighted_centroids)
    parts = [
        local_density_features(centroids, radius=cfg.density_radius),
        neighbor_graph_signature(centroids, neighbor_k=cfg.neighbor_k),
    ]
    fov = getattr(plane, "fov", None)
    if fov is not None:
        parts.append(
            fov_patch_moment_descriptors(
                np.asarray(fov, dtype=float),
                centroids,
                patch_radius=cfg.patch_radius,
                histogram_bins=cfg.histogram_bins,
            )
        )
    return (
        np.column_stack(parts).astype(float)
        if parts
        else np.zeros((centroids.shape[0], 0), dtype=float)
    )


def pairwise_context_distance(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    order: str = "xy",
    weighted_centroids: bool = False,
    config: ContextDescriptorConfig | None = None,
) -> np.ndarray:
    """Return robust-normalized pairwise descriptor distances."""

    ref = roi_context_descriptors(
        reference_plane,
        order=order,
        weighted_centroids=weighted_centroids,
        config=config,
    )
    meas = roi_context_descriptors(
        measurement_plane,
        order=order,
        weighted_centroids=weighted_centroids,
        config=config,
    )
    if ref.shape[0] == 0 or meas.shape[0] == 0:
        return np.zeros((ref.shape[0], meas.shape[0]), dtype=float)
    pooled = np.vstack((ref, meas))
    center = np.nanmedian(pooled, axis=0)
    q75, q25 = np.nanpercentile(pooled, [75.0, 25.0], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(pooled, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    ref_norm = np.nan_to_num(
        (ref - center) / scale, nan=0.0, posinf=1.0e6, neginf=-1.0e6
    )
    meas_norm = np.nan_to_num(
        (meas - center) / scale, nan=0.0, posinf=1.0e6, neginf=-1.0e6
    )
    diffs = ref_norm[:, None, :] - meas_norm[None, :, :]
    return np.sqrt(np.mean(diffs * diffs, axis=2))


def local_density_descriptor(centroids_xy: Any, *, radius: float) -> np.ndarray:
    """Return the number of neighboring ROIs within ``radius`` for each centroid."""

    return local_density_features(centroids_xy, radius=radius)[:, 0]


def pairwise_context_components(
    reference_centroids_xy: Any,
    measurement_centroids_xy: Any,
    *,
    config: dict[str, Any] | ContextDescriptorConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return pairwise local-context cost components for centroid arrays."""

    density_radius = (
        config.density_radius
        if isinstance(config, ContextDescriptorConfig)
        else float(
            (config or {}).get(
                "density_radius", ContextDescriptorConfig().density_radius
            )
        )
    )
    reference_density = local_density_descriptor(
        reference_centroids_xy, radius=density_radius
    )
    measurement_density = local_density_descriptor(
        measurement_centroids_xy, radius=density_radius
    )
    return {
        "local_density_cost": np.abs(
            reference_density[:, None] - measurement_density[None, :]
        )
    }


def local_density_features(centroids_xy: Any, *, radius: float) -> np.ndarray:
    """Return local ROI-density and nearest-neighbor summary features."""

    centroids = np.asarray(centroids_xy, dtype=float)
    n = centroids.shape[0]
    if n == 0:
        return np.zeros((0, 4), dtype=float)
    distances = _pairwise_distances(centroids)
    np.fill_diagonal(distances, np.inf)
    density = np.sum(distances <= float(radius), axis=1).astype(float)
    nearest = np.min(distances, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, 0.0)
    finite_distances = np.where(np.isfinite(distances), distances, np.nan)
    median_neighbor = np.nanmedian(finite_distances, axis=1)
    median_neighbor = np.nan_to_num(median_neighbor, nan=0.0)
    crowding = density / max(float(n - 1), 1.0)
    return np.column_stack((density, crowding, nearest, median_neighbor))


def neighbor_graph_signature(centroids_xy: Any, *, neighbor_k: int) -> np.ndarray:
    """Return sorted normalized neighbor-distance signatures per ROI."""

    centroids = np.asarray(centroids_xy, dtype=float)
    n = centroids.shape[0]
    signature = np.zeros((n, int(neighbor_k)), dtype=float)
    if n <= 1:
        return signature
    distances = _pairwise_distances(centroids)
    np.fill_diagonal(distances, np.inf)
    finite = distances[np.isfinite(distances)]
    scale = float(np.median(finite)) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    for roi_index in range(n):
        row = np.sort(distances[roi_index])
        row = row[np.isfinite(row)]
        used = min(int(neighbor_k), row.size)
        if used:
            signature[roi_index, :used] = row[:used] / scale
            if used < neighbor_k:
                signature[roi_index, used:] = signature[roi_index, used - 1]
    return signature


def fov_patch_moments(
    image: Any, centroids_xy: Any, *, patch_radius: int
) -> np.ndarray:
    """Return mean and standard-deviation moments for local FOV patches."""

    img = np.asarray(image, dtype=float)
    centroids = np.asarray(centroids_xy, dtype=float)
    moments = np.zeros((centroids.shape[0], 2), dtype=float)
    if img.ndim != 2:
        return moments
    for roi_index, (x_coord, y_coord) in enumerate(centroids):
        patch = _crop_patch(img, x_coord, y_coord, radius=patch_radius)
        values = patch[np.isfinite(patch)]
        if values.size:
            moments[roi_index, 0] = float(np.mean(values))
            moments[roi_index, 1] = float(np.std(values))
    return moments


def fov_patch_moment_descriptors(
    image: Any,
    centroids_xy: Any,
    *,
    patch_radius: int,
    histogram_bins: int,
) -> np.ndarray:
    """Return local FOV patch moments and histogram descriptors."""

    img = np.asarray(image, dtype=float)
    centroids = np.asarray(centroids_xy, dtype=float)
    n = centroids.shape[0]
    width = 6 + int(histogram_bins)
    descriptors = np.zeros((n, width), dtype=float)
    if img.ndim != 2:
        return descriptors
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return descriptors
    global_min = float(np.min(finite))
    global_max = float(np.max(finite))
    bin_edges = np.linspace(global_min, global_max + 1.0e-12, int(histogram_bins) + 1)
    filled_img = np.nan_to_num(img, nan=float(np.mean(finite)))
    if min(img.shape) < 2:
        grad_mag = np.zeros_like(filled_img, dtype=float)
    else:
        grad_y, grad_x = np.gradient(filled_img)
        grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    for roi_index, (x_coord, y_coord) in enumerate(centroids):
        patch = _crop_patch(img, x_coord, y_coord, radius=patch_radius)
        grad_patch = _crop_patch(grad_mag, x_coord, y_coord, radius=patch_radius)
        values = patch[np.isfinite(patch)]
        if values.size == 0:
            continue
        hist, _ = np.histogram(values, bins=bin_edges)
        hist = hist.astype(float) / max(float(np.sum(hist)), 1.0)
        descriptors[roi_index, 0] = float(np.mean(values))
        descriptors[roi_index, 1] = float(np.std(values))
        descriptors[roi_index, 2] = float(np.percentile(values, 10.0))
        descriptors[roi_index, 3] = float(np.percentile(values, 90.0))
        descriptors[roi_index, 4] = (
            float(np.mean(grad_patch[np.isfinite(grad_patch)]))
            if np.any(np.isfinite(grad_patch))
            else 0.0
        )
        descriptors[roi_index, 5] = float(values.size)
        descriptors[roi_index, 6:] = hist
    return descriptors


def _integer_control(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(numeric_value)
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _integer_at_least(
    value: Any,
    *,
    minimum: int,
    name: str,
    message: str,
) -> int:
    integer_value = _integer_control(value, name=name)
    if integer_value < minimum:
        raise ValueError(message)
    return integer_value


def _nonnegative_int(value: Any, *, name: str) -> int:
    return _integer_at_least(
        value,
        minimum=0,
        name=name,
        message=f"{name} must be non-negative",
    )


def _positive_int(value: Any, *, name: str) -> int:
    return _integer_at_least(
        value,
        minimum=1,
        name=name,
        message=f"{name} must be positive",
    )


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(f"{name} must be positive")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric_value


def _centroids_xy(plane: Any, *, order: str, weighted: bool) -> np.ndarray:
    centroids = np.asarray(plane.centroids(order=order, weighted=weighted), dtype=float)
    if order == "xy":
        return centroids.T
    return centroids[[1, 0], :].T


def _pairwise_distances(points_xy: np.ndarray) -> np.ndarray:
    diffs = points_xy[:, None, :] - points_xy[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def _crop_patch(
    image: np.ndarray, x_coord: float, y_coord: float, *, radius: int
) -> np.ndarray:
    height, width = image.shape
    x = int(round(float(x_coord)))
    y = int(round(float(y_coord)))
    y0 = max(y - radius, 0)
    y1 = min(y + radius + 1, height)
    x0 = max(x - radius, 0)
    x1 = min(x + radius + 1, width)
    if y0 >= y1 or x0 >= x1:
        return np.zeros((0, 0), dtype=float)
    return np.asarray(image[y0:y1, x0:x1], dtype=float)
