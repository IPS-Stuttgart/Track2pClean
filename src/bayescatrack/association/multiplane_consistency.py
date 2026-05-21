"""Cross-plane consistency helpers for registration and tracking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MultiPlaneConsistencyConfig:
    """Weights for sharing registration-quality evidence across planes."""

    shared_shift_weight: float = 0.25
    shared_quality_weight: float = 0.50
    min_plane_count: int = 2

    def __post_init__(self) -> None:
        if self.shared_shift_weight < 0.0 or self.shared_quality_weight < 0.0:
            raise ValueError("weights must be non-negative")
        if self.min_plane_count < 1:
            raise ValueError("min_plane_count must be positive")


@dataclass(frozen=True)
class PlaneRegistrationQuality:
    """Minimal registration-quality summary for one imaging plane."""

    plane_name: str
    registration_rmse: float
    valid_fraction: float = 1.0


def shared_registration_reliability(
    qualities: Sequence[PlaneRegistrationQuality],
) -> float:
    """Return a bounded reliability score shared across plane registrations."""

    if not qualities:
        return 0.0
    rmse = np.asarray([quality.registration_rmse for quality in qualities], dtype=float)
    valid = np.asarray([quality.valid_fraction for quality in qualities], dtype=float)
    rmse_score = 1.0 / (1.0 + max(float(np.nanmean(np.maximum(rmse, 0.0))), 0.0))
    valid_score = float(np.nanmean(np.clip(valid, 0.0, 1.0)))
    return float(np.clip(rmse_score * valid_score, 0.0, 1.0))


def apply_multiplane_quality_penalty(
    cost_matrix: Any,
    qualities: Sequence[PlaneRegistrationQuality],
    *,
    penalty_weight: float = 1.0,
) -> np.ndarray:
    """Add a shared penalty when registration quality is unreliable."""

    reliability = shared_registration_reliability(qualities)
    penalty = max(1.0 - reliability, 0.0) * float(penalty_weight)
    return np.asarray(cost_matrix, dtype=float) + penalty


def aggregate_registration_metadata_by_edge(
    plane_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Aggregate registration metadata observed across imaging planes."""

    shifts_y = _finite_values(plane_metadata, ("fov_translation_shift_y", "shift_y"))
    shifts_x = _finite_values(plane_metadata, ("fov_translation_shift_x", "shift_x"))
    peaks = _finite_values(
        plane_metadata, ("fov_translation_peak_correlation", "peak_correlation")
    )
    rmses = _finite_values(
        plane_metadata,
        ("fov_affine_fit_rmse", "fit_rmse", "nonrigid_registration_fit_rmse"),
    )
    return {
        "plane_count": int(len(plane_metadata)),
        "median_shift_y": _median_or_nan(shifts_y),
        "median_shift_x": _median_or_nan(shifts_x),
        "shift_y_mad": _mad_or_nan(shifts_y),
        "shift_x_mad": _mad_or_nan(shifts_x),
        "median_peak_correlation": _median_or_nan(peaks),
        "median_fit_rmse": _median_or_nan(rmses),
    }


def shared_registration_quality_penalty(
    metadata_summary: Mapping[str, Any],
    *,
    config: MultiPlaneConsistencyConfig | None = None,
) -> float:
    """Return a scalar penalty for plane-to-plane registration inconsistency."""

    cfg = config or MultiPlaneConsistencyConfig()
    plane_count = int(metadata_summary.get("plane_count", 0))
    if plane_count < cfg.min_plane_count:
        return 0.0
    shift_mad = _safe_float(metadata_summary.get("shift_y_mad"), 0.0) + _safe_float(
        metadata_summary.get("shift_x_mad"), 0.0
    )
    rmse = _safe_float(metadata_summary.get("median_fit_rmse"), 0.0)
    peak = _safe_float(metadata_summary.get("median_peak_correlation"), 1.0)
    return float(
        cfg.shared_shift_weight * shift_mad
        + cfg.shared_quality_weight * max(rmse, 0.0)
        + cfg.shared_quality_weight * max(1.0 - peak, 0.0)
    )


def adjust_plane_costs_by_shared_quality(
    cost_matrices: Sequence[Any],
    metadata_by_plane: Sequence[Mapping[str, Any]],
    *,
    config: MultiPlaneConsistencyConfig | None = None,
) -> list[np.ndarray]:
    """Add a shared consistency penalty to each plane's edge costs."""

    summary = aggregate_registration_metadata_by_edge(metadata_by_plane)
    penalty = shared_registration_quality_penalty(summary, config=config)
    return [np.asarray(costs, dtype=float) + penalty for costs in cost_matrices]


def _finite_values(
    rows: Sequence[Mapping[str, Any]], names: tuple[str, ...]
) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        for name in names:
            if name not in row:
                continue
            value = _safe_float(row.get(name), np.nan)
            if np.isfinite(value):
                values.append(value)
                break
    return np.asarray(values, dtype=float)


def _median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def _mad_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _safe_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if np.isfinite(numeric) else float(default)
