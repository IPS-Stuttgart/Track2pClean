"""Observable-context intercepts for calibrated association probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SessionContext:
    """Subject/session-edge covariates that are observable without GT labels."""

    session_gap: float = 1.0
    roi_density: float = 0.0
    mean_cell_probability: float = 0.5
    registration_fit_rmse: float = 0.0
    registration_valid_fraction: float = 1.0
    trace_availability_fraction: float = 0.0
    backend_bias: float = 0.0


@dataclass(frozen=True)
class AdaptiveCalibrationConfig:
    """Weights for a simple hierarchical/intercept-style calibration correction."""

    base_intercept: float = 0.0
    session_gap_weight: float = -0.15
    roi_density_weight: float = -0.10
    low_cell_probability_weight: float = -0.50
    registration_rmse_weight: float = -0.05
    invalid_warp_weight: float = -1.00
    trace_available_weight: float = 0.10
    max_abs_intercept: float = 5.0

    def __post_init__(self) -> None:
        if self.max_abs_intercept <= 0.0:
            raise ValueError("max_abs_intercept must be positive")


def session_context_from_planes(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    session_gap: int | float = 1.0,
    registration_metadata: Mapping[str, Any] | None = None,
) -> SessionContext:
    """Build observable calibration context from two planes and registration metadata."""

    image_shape = getattr(reference_plane, "image_shape", (1, 1))
    image_area = max(float(image_shape[0] * image_shape[1]), 1.0)
    roi_density = (
        float(
            getattr(reference_plane, "n_rois", 0)
            + getattr(measurement_plane, "n_rois", 0)
        )
        / image_area
    )
    probabilities = []
    for plane in (reference_plane, measurement_plane):
        cell_probs = getattr(plane, "cell_probabilities", None)
        if cell_probs is not None:
            arr = np.asarray(cell_probs, dtype=float)
            if arr.size:
                probabilities.append(float(np.nanmean(arr)))
    mean_cell_probability = float(np.mean(probabilities)) if probabilities else 0.5
    trace_available = [
        getattr(plane, "traces", None) is not None
        or getattr(plane, "spike_traces", None) is not None
        for plane in (reference_plane, measurement_plane)
    ]
    metadata = {} if registration_metadata is None else dict(registration_metadata)
    return SessionContext(
        session_gap=float(session_gap),
        roi_density=roi_density,
        mean_cell_probability=mean_cell_probability,
        registration_fit_rmse=_first_scalar(
            metadata,
            ("fit_rmse", "fov_affine_fit_rmse", "nonrigid_registration_fit_rmse"),
            default=0.0,
        ),
        registration_valid_fraction=_first_scalar(
            metadata,
            ("valid_fraction", "nonrigid_registration_inverse_warp_valid_fraction"),
            default=1.0,
        ),
        trace_availability_fraction=float(np.mean(trace_available)),
        backend_bias=_backend_bias(metadata),
    )


def context_intercept(
    context: SessionContext,
    *,
    config: AdaptiveCalibrationConfig | None = None,
) -> float:
    """Return a bounded logit intercept from observable context."""

    cfg = config or AdaptiveCalibrationConfig()
    gap_excess = max(float(context.session_gap) - 1.0, 0.0)
    low_cell_probability = max(1.0 - float(context.mean_cell_probability), 0.0)
    invalid_fraction = max(1.0 - float(context.registration_valid_fraction), 0.0)
    intercept = (
        cfg.base_intercept
        + cfg.session_gap_weight * gap_excess
        + cfg.roi_density_weight * float(context.roi_density)
        + cfg.low_cell_probability_weight * low_cell_probability
        + cfg.registration_rmse_weight * max(float(context.registration_fit_rmse), 0.0)
        + cfg.invalid_warp_weight * invalid_fraction
        + cfg.trace_available_weight * float(context.trace_availability_fraction)
        + float(context.backend_bias)
    )
    return float(np.clip(intercept, -cfg.max_abs_intercept, cfg.max_abs_intercept))


def apply_context_intercept_to_probabilities(
    probabilities: Any,
    context: SessionContext,
    *,
    config: AdaptiveCalibrationConfig | None = None,
) -> np.ndarray:
    """Shift calibrated probabilities in logit space using observable context."""

    p = np.asarray(probabilities, dtype=float)
    logits = _logit(np.clip(p, 1.0e-9, 1.0 - 1.0e-9))
    return _sigmoid(logits + context_intercept(context, config=config))


def probability_cost_matrix(
    probabilities: Any, *, epsilon: float = 1.0e-9
) -> np.ndarray:
    """Convert match probabilities into non-negative assignment costs."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return -np.log(np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0))


def apply_context_intercept_to_costs(
    cost_matrix: Any,
    context: SessionContext,
    *,
    config: AdaptiveCalibrationConfig | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Approximate a context logit shift directly on costs."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    costs = np.asarray(cost_matrix, dtype=float)
    return costs - float(temperature) * context_intercept(context, config=config)


def _first_scalar(
    metadata: Mapping[str, Any], names: tuple[str, ...], *, default: float
) -> float:
    for name in names:
        if name not in metadata:
            continue
        try:
            value = float(np.asarray(metadata[name]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(value):
            return value
    return float(default)


def _backend_bias(metadata: Mapping[str, Any]) -> float:
    backend = str(metadata.get("registration_backend", "")).lower()
    if "nonrigid" in backend or "bspline" in backend or "tps" in backend:
        return 0.05
    if "translation" in backend:
        return -0.05
    return 0.0


def _logit(probabilities: np.ndarray) -> np.ndarray:
    return np.log(probabilities / (1.0 - probabilities))


def _sigmoid(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr, dtype=float)
    positive = arr >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_values = np.exp(arr[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out
