"""Uncertainty-aware utilities for Track2p-style ROI association.

These helpers keep the existing deterministic cost-matrix pipeline intact while
adding a second, explicit reliability layer.  The intent is to avoid treating a
low-cost edge from a dubious registration, empty warped mask, or highly ambiguous
local competition as equally trustworthy as a low-cost edge from clean evidence.

The module is deliberately dependency-light and can be used from benchmark
experiments without changing PyRecEst.  It operates on the same pairwise cost
matrices and component dictionaries already produced by BayesCaTrack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class EdgeUncertaintyConfig:
    """Weights for converting diagnostics into edge reliability and cost penalties.

    ``temperature`` controls the cost-to-probability softmax scale.  Reliability
    weights are additive in log space, so every unreliable cue increases the
    uncertainty penalty by ``weight * normalized_component``.
    """

    temperature: float = 2.0
    uncertainty_penalty_weight: float = 1.0
    registration_rmse_weight: float = 0.25
    invalid_warp_fraction_weight: float = 2.0
    empty_registered_roi_weight: float = 8.0
    gated_edge_weight: float = 8.0
    covariance_logdet_weight: float = 0.10
    local_margin_weight: float = 0.75
    activity_missing_weight: float = 0.25
    min_reliability: float = 1.0e-6
    max_penalty: float = 1.0e6

    def __post_init__(self) -> None:
        if _finite_float_value(self.temperature, "temperature") <= 0.0:
            raise ValueError("temperature must be positive")
        for name in (
            "uncertainty_penalty_weight",
            "registration_rmse_weight",
            "invalid_warp_fraction_weight",
            "empty_registered_roi_weight",
            "gated_edge_weight",
            "covariance_logdet_weight",
            "local_margin_weight",
            "activity_missing_weight",
        ):
            if _finite_float_value(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        min_reliability = _finite_float_value(
            self.min_reliability, "min_reliability"
        )
        if min_reliability <= 0.0 or min_reliability > 1.0:
            raise ValueError("min_reliability must lie in (0, 1]")
        if _finite_float_value(self.max_penalty, "max_penalty") <= 0.0:
            raise ValueError("max_penalty must be positive")


@dataclass(frozen=True)
class UncertaintyAwareEdgeResult:
    """Cost/probability matrices after uncertainty adjustment."""

    adjusted_cost_matrix: np.ndarray
    posterior_probability_matrix: np.ndarray
    reliability_matrix: np.ndarray
    uncertainty_penalty_matrix: np.ndarray


def edge_uncertainty_config_from_mapping(
    value: EdgeUncertaintyConfig | Mapping[str, Any] | None,
) -> EdgeUncertaintyConfig | None:
    """Normalize optional uncertainty configuration values.

    Benchmark manifests and CLI JSON arguments naturally pass dictionaries,
    while programmatic callers may pass an already-instantiated config. Keeping
    the normalizer here avoids every benchmark/association entry point growing
    its own slightly different coercion logic.
    """

    if value is None:
        return None
    if isinstance(value, EdgeUncertaintyConfig):
        return value
    return EdgeUncertaintyConfig(**dict(value))


def _finite_float_value(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def uncertainty_aware_cost_matrix(
    cost_matrix: Any,
    pairwise_components: Mapping[str, Any] | None = None,
    *,
    registration_metadata: Mapping[str, Any] | None = None,
    empty_registered_rois: Any | None = None,
    config: EdgeUncertaintyConfig | None = None,
) -> UncertaintyAwareEdgeResult:
    """Return costs and pseudo-posteriors adjusted by reliability diagnostics.

    Parameters
    ----------
    cost_matrix
        Base deterministic assignment costs, lower is better.
    pairwise_components
        Optional BayesCaTrack pairwise component matrices.  Components such as
        ``gated``, ``covariance_logdet_cost``, ``activity_*_available``, and
        ``centroid_rank_cost`` are consumed when present.
    registration_metadata
        Optional scalar metadata from a registration result.  Recognized keys
        include ``fit_rmse``, ``fov_affine_fit_rmse``,
        ``nonrigid_registration_fit_rmse``, ``valid_fraction`` and
        ``nonrigid_registration_inverse_warp_valid_fraction``.
    empty_registered_rois
        Optional boolean vector with one entry per measurement ROI.  Empty warped
        masks are downweighted/penalized column-wise.
    config
        Reliability and penalty weights.
    """

    cfg = config or EdgeUncertaintyConfig()
    costs = _as_cost_matrix(cost_matrix)
    components = {} if pairwise_components is None else pairwise_components
    reliability = edge_reliability_matrix(
        costs.shape,
        components,
        registration_metadata=registration_metadata,
        empty_registered_rois=empty_registered_rois,
        config=cfg,
    )
    penalty = cfg.uncertainty_penalty_weight * (-np.log(reliability))
    adjusted = _finite_costs(costs + penalty, large_cost=cfg.max_penalty)
    posterior = posterior_probability_matrix(
        adjusted,
        reliability_matrix=reliability,
        temperature=cfg.temperature,
    )
    return UncertaintyAwareEdgeResult(
        adjusted_cost_matrix=adjusted,
        posterior_probability_matrix=posterior,
        reliability_matrix=reliability,
        uncertainty_penalty_matrix=penalty,
    )


def edge_reliability_matrix(
    shape: tuple[int, int],
    pairwise_components: Mapping[str, Any] | None = None,
    *,
    registration_metadata: Mapping[str, Any] | None = None,
    empty_registered_rois: Any | None = None,
    config: EdgeUncertaintyConfig | None = None,
) -> np.ndarray:
    """Return a multiplicative reliability matrix in ``[min_reliability, 1]``."""

    cfg = config or EdgeUncertaintyConfig()
    n_reference, n_measurement = int(shape[0]), int(shape[1])
    penalty = np.zeros((n_reference, n_measurement), dtype=float)
    components = {} if pairwise_components is None else pairwise_components

    gated = _optional_component(components, "gated", shape)
    if gated is not None:
        penalty += cfg.gated_edge_weight * (np.asarray(gated, dtype=float) > 0.0)

    covariance_logdet = _optional_component(components, "covariance_logdet_cost", shape)
    if covariance_logdet is not None:
        penalty += cfg.covariance_logdet_weight * _robust_unit_scale(covariance_logdet)

    centroid_rank = _optional_component(components, "centroid_rank_cost", shape)
    if centroid_rank is not None:
        penalty += cfg.local_margin_weight * np.clip(
            np.asarray(centroid_rank, dtype=float), 0.0, 1.0
        )

    # Missing activity should not dominate a decision, but it is useful as a weak
    # indicator that calibrated probabilities are less trustworthy.
    for availability_name in (
        "activity_similarity_available",
        "activity_tiebreaker_available",
        "fluorescence_similarity_available",
        "spike_similarity_available",
    ):
        availability = _optional_component(components, availability_name, shape)
        if availability is not None:
            penalty += cfg.activity_missing_weight * (
                1.0 - np.clip(np.asarray(availability, dtype=float), 0.0, 1.0)
            )
            break

    if empty_registered_rois is not None:
        empty = _column_mask_for_cost_shape(empty_registered_rois, shape)
        penalty[:, empty] += cfg.empty_registered_roi_weight

    metadata = {} if registration_metadata is None else dict(registration_metadata)
    rmse = _first_finite_scalar(
        metadata,
        (
            "fit_rmse",
            "fov_affine_fit_rmse",
            "nonrigid_registration_fit_rmse",
            "registration_fit_rmse",
        ),
    )
    if rmse is not None:
        penalty += cfg.registration_rmse_weight * max(float(rmse), 0.0)

    valid_fraction = _first_finite_scalar(
        metadata,
        (
            "valid_fraction",
            "nonrigid_registration_inverse_warp_valid_fraction",
            "registration_valid_fraction",
        ),
    )
    if valid_fraction is not None:
        invalid_fraction = 1.0 - float(np.clip(valid_fraction, 0.0, 1.0))
        penalty += cfg.invalid_warp_fraction_weight * invalid_fraction

    reliability = np.exp(-np.clip(penalty, 0.0, 100.0))
    return np.clip(reliability, cfg.min_reliability, 1.0)


def posterior_probability_matrix(
    cost_matrix: Any,
    *,
    reliability_matrix: Any | None = None,
    temperature: float = 2.0,
) -> np.ndarray:
    """Return a row-wise pseudo-posterior probability matrix from costs.

    The probabilities are not a replacement for supervised calibration.  They are
    a stable diagnostic/post-processing quantity: lower costs and higher
    reliability yield higher row-wise probabilities.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    costs = _as_cost_matrix(cost_matrix)
    finite = np.isfinite(costs)
    shifted = np.where(finite, costs, np.inf)
    row_min = np.min(np.where(finite, shifted, np.inf), axis=1, keepdims=True)
    row_min = np.where(np.isfinite(row_min), row_min, 0.0)
    logits = -np.where(finite, costs - row_min, np.inf) / float(temperature)
    scores = np.where(finite, np.exp(np.clip(logits, -100.0, 50.0)), 0.0)
    if reliability_matrix is not None:
        reliability = np.asarray(reliability_matrix, dtype=float)
        if reliability.shape != costs.shape:
            raise ValueError("reliability_matrix must match cost_matrix shape")
        scores *= np.clip(reliability, 0.0, 1.0)
    row_sums = np.sum(scores, axis=1, keepdims=True)
    probabilities = np.zeros_like(scores, dtype=float)
    np.divide(scores, row_sums, out=probabilities, where=row_sums > 0.0)
    return probabilities


def candidate_mask_from_posteriors(
    probabilities: Any,
    *,
    min_probability: float = 0.0,
    row_top_k: int | None = None,
    column_top_k: int | None = None,
) -> np.ndarray:
    """Return a candidate mask from probability threshold and row/column top-k."""

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix")
    if min_probability < 0.0 or min_probability > 1.0:
        raise ValueError("min_probability must lie in [0, 1]")
    mask = probs >= float(min_probability)
    if row_top_k is not None:
        mask |= _top_k_mask(probs, axis=1, k=int(row_top_k))
    if column_top_k is not None:
        mask |= _top_k_mask(probs, axis=0, k=int(column_top_k))
    return mask


def apply_candidate_mask(
    cost_matrix: Any,
    candidate_mask: Any,
    *,
    large_cost: float = 1.0e6,
) -> np.ndarray:
    """Mask rejected candidate links by assigning a finite large cost."""

    costs = _as_cost_matrix(cost_matrix)
    mask = np.asarray(candidate_mask, dtype=bool)
    if mask.shape != costs.shape:
        raise ValueError("candidate_mask must match cost_matrix shape")
    return np.where(mask, costs, float(large_cost))


def _top_k_mask(values: np.ndarray, *, axis: int, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("top-k values must be positive")
    values = np.asarray(values, dtype=float)
    mask = np.zeros(values.shape, dtype=bool)
    if axis == 1:
        for row_index, row in enumerate(values):
            finite_indices = np.flatnonzero(np.isfinite(row))
            if finite_indices.size:
                order = finite_indices[np.argsort(row[finite_indices])[::-1]]
                mask[row_index, order[:k]] = True
    elif axis == 0:
        for col_index, col in enumerate(values.T):
            finite_indices = np.flatnonzero(np.isfinite(col))
            if finite_indices.size:
                order = finite_indices[np.argsort(col[finite_indices])[::-1]]
                mask[order[:k], col_index] = True
    else:
        raise ValueError("axis must be 0 or 1")
    return mask


def _optional_component(
    components: Mapping[str, Any], name: str, shape: tuple[int, int]
) -> np.ndarray | None:
    if name not in components:
        return None
    values = np.asarray(components[name], dtype=float)
    if values.shape != shape:
        return None
    return np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def _as_cost_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    return matrix


def _finite_costs(values: np.ndarray, *, large_cost: float) -> np.ndarray:
    return np.nan_to_num(values, nan=large_cost, posinf=large_cost, neginf=large_cost)


def _robust_unit_scale(values: Any) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=1.0e6)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=float)
    q90 = float(np.percentile(finite, 90.0))
    scale = q90 if np.isfinite(q90) and q90 > 1.0e-12 else 1.0
    return np.clip(array / scale, 0.0, 10.0)


def _column_mask_for_cost_shape(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    """Return a registered-ROI column mask aligned to compact/full costs."""

    column_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if column_mask.shape == (shape[1],):
        return column_mask
    compact_column_count = int(column_mask.size - np.count_nonzero(column_mask))
    if compact_column_count == shape[1]:
        return np.zeros((shape[1],), dtype=bool)
    raise ValueError(
        "empty_registered_rois must have one entry per compact column or one "
        "entry per original measurement ROI"
    )


def _first_finite_scalar(
    metadata: Mapping[str, Any], candidate_names: tuple[str, ...]
) -> float | None:
    for name in candidate_names:
        if name not in metadata:
            continue
        try:
            value = float(np.asarray(metadata[name]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(value):
            return value
    return None
