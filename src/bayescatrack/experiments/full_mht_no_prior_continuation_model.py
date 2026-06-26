"""Label-free no-prior continuation likelihood for FullMHT.

The calibrated association probe exposed a specific failure mode: when Track2p has
no proposal successor for a source ROI, the local calibrated edge score can still
open long chains of non-prior continuations.  A scalar death/continuation penalty
controls that failure only brittlely.

This module turns no-prior continuation diagnostics into a robust likelihood
ratio.  It uses high-confidence, geometry/growth-consistent candidate edges as
pseudo-continuation anchors and weak local candidates as the death/background
reference.  It does not read benchmark labels or audit columns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "registered_iou",
    "shifted_iou",
    "min_cell_probability",
    "area_similarity",
    "threshold_margin",
    "negative_centroid_distance",
    "negative_growth_residual",
    "negative_growth_mahalanobis",
    "negative_local_deformation",
    "negative_log_row_rank",
    "negative_log_column_rank",
)


@dataclass(frozen=True)
class NoPriorContinuationDiagnostics:
    """Label-free diagnostics for one no-prior continuation candidate."""

    registered_iou: float
    shifted_iou: float
    growth_residual: float
    growth_mahalanobis: float
    min_cell_probability: float
    area_ratio: float = 1.0
    centroid_distance: float = 0.0
    threshold_margin: float = 0.0
    local_deformation: float = 0.0
    row_rank: int = 1
    column_rank: int = 1


@dataclass(frozen=True)
class NoPriorContinuationConfig:
    """Pseudo-label and likelihood controls for no-prior continuations."""

    min_anchor_registered_iou: float = 0.70
    min_anchor_shifted_iou: float = 0.55
    max_anchor_growth_mahalanobis: float = 1.5
    max_anchor_growth_residual: float = 1.5
    min_anchor_cell_probability: float = 0.80
    max_anchor_local_deformation: float = 0.15
    max_anchor_rank: int = 1
    max_background_registered_iou: float = 0.45
    max_background_shifted_iou: float = 0.45
    min_background_growth_mahalanobis: float = 2.5
    min_background_growth_residual: float = 2.0
    max_background_cell_probability: float = 0.70
    min_background_local_deformation: float = 0.30
    min_examples_per_class: int = 2
    min_feature_scale: float = 0.05
    per_feature_clip: float = 4.0
    score_clip: float = 8.0


@dataclass(frozen=True)
class NoPriorContinuationModel:
    """Robust Gaussian likelihood-ratio model for no-prior continuations."""

    feature_names: tuple[str, ...]
    continuation_location: np.ndarray
    continuation_scale: np.ndarray
    background_location: np.ndarray
    background_scale: np.ndarray
    used_features: np.ndarray
    per_feature_clip: float
    score_clip: float

    @property
    def enabled(self) -> bool:
        return bool(np.any(np.asarray(self.used_features, dtype=bool)))

    def log_continuation_ratio(
        self, diagnostics: Sequence[NoPriorContinuationDiagnostics]
    ) -> np.ndarray:
        """Return positive-for-continuation log-likelihood ratios."""

        features = no_prior_continuation_feature_matrix(diagnostics)
        if features.shape[1] != len(self.feature_names):
            raise ValueError("Feature dimension mismatch for continuation model")
        if not self.enabled:
            return np.zeros(features.shape[0], dtype=float)
        continuation_ll = _gaussian_log_density(
            features,
            location=np.asarray(self.continuation_location, dtype=float),
            scale=np.asarray(self.continuation_scale, dtype=float),
        )
        background_ll = _gaussian_log_density(
            features,
            location=np.asarray(self.background_location, dtype=float),
            scale=np.asarray(self.background_scale, dtype=float),
        )
        llr = continuation_ll - background_ll
        used = np.asarray(self.used_features, dtype=bool)
        llr[:, ~used] = 0.0
        if float(self.per_feature_clip) > 0.0:
            llr = np.clip(
                llr,
                -float(self.per_feature_clip),
                float(self.per_feature_clip),
            )
        denom = max(1.0, math.sqrt(float(np.sum(used))))
        scores = np.sum(llr, axis=1) / denom
        if float(self.score_clip) > 0.0:
            scores = np.clip(scores, -float(self.score_clip), float(self.score_clip))
        return scores.astype(float)


def calibrate_no_prior_continuation_model(
    diagnostics: Sequence[NoPriorContinuationDiagnostics],
    *,
    config: NoPriorContinuationConfig | None = None,
) -> NoPriorContinuationModel:
    """Fit a label-free no-prior continuation likelihood-ratio model."""

    cfg = config or NoPriorContinuationConfig()
    features = no_prior_continuation_feature_matrix(diagnostics)
    anchor_mask = pseudo_continuation_anchor_mask(diagnostics, config=cfg)
    background_mask = pseudo_death_background_mask(diagnostics, config=cfg)
    n_features = len(FEATURE_NAMES)
    empty = _empty_model(cfg)
    if features.shape[0] == 0 or features.shape[1] != n_features:
        return empty
    if int(np.sum(anchor_mask)) < int(cfg.min_examples_per_class):
        return empty
    if int(np.sum(background_mask)) < int(cfg.min_examples_per_class):
        return empty

    continuation = features[np.asarray(anchor_mask, dtype=bool)]
    background = features[np.asarray(background_mask, dtype=bool)]
    continuation_location = np.zeros(n_features, dtype=float)
    continuation_scale = np.ones(n_features, dtype=float)
    background_location = np.zeros(n_features, dtype=float)
    background_scale = np.ones(n_features, dtype=float)
    used = np.zeros(n_features, dtype=bool)
    for index in range(n_features):
        continuation_values = _finite_column(continuation[:, index])
        background_values = _finite_column(background[:, index])
        if continuation_values.size < int(cfg.min_examples_per_class):
            continue
        if background_values.size < int(cfg.min_examples_per_class):
            continue
        continuation_location[index], continuation_scale[index] = _robust_location_scale(
            continuation_values,
            min_scale=float(cfg.min_feature_scale),
        )
        background_location[index], background_scale[index] = _robust_location_scale(
            background_values,
            min_scale=float(cfg.min_feature_scale),
        )
        if continuation_location[index] > background_location[index] + 1.0e-9:
            used[index] = True
    if not np.any(used):
        return empty
    return NoPriorContinuationModel(
        feature_names=FEATURE_NAMES,
        continuation_location=continuation_location,
        continuation_scale=continuation_scale,
        background_location=background_location,
        background_scale=background_scale,
        used_features=used,
        per_feature_clip=float(cfg.per_feature_clip),
        score_clip=float(cfg.score_clip),
    )


def no_prior_continuation_feature_matrix(
    diagnostics: Sequence[NoPriorContinuationDiagnostics],
) -> np.ndarray:
    """Convert no-prior continuation diagnostics to a numeric feature matrix."""

    rows = [_feature_row(item) for item in diagnostics]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    return np.asarray(rows, dtype=float).reshape(len(rows), len(FEATURE_NAMES))


def pseudo_continuation_anchor_mask(
    diagnostics: Sequence[NoPriorContinuationDiagnostics],
    *,
    config: NoPriorContinuationConfig | None = None,
) -> np.ndarray:
    """Return high-confidence label-free pseudo-continuation anchors."""

    cfg = config or NoPriorContinuationConfig()
    return np.asarray(
        [
            _finite(item.registered_iou) >= float(cfg.min_anchor_registered_iou)
            and _finite(item.shifted_iou) >= float(cfg.min_anchor_shifted_iou)
            and _finite(item.growth_mahalanobis) <= float(
                cfg.max_anchor_growth_mahalanobis
            )
            and _finite(item.growth_residual) <= float(cfg.max_anchor_growth_residual)
            and _finite(item.min_cell_probability) >= float(
                cfg.min_anchor_cell_probability
            )
            and _finite(item.local_deformation) <= float(
                cfg.max_anchor_local_deformation
            )
            and int(item.row_rank) <= int(cfg.max_anchor_rank)
            and int(item.column_rank) <= int(cfg.max_anchor_rank)
            for item in diagnostics
        ],
        dtype=bool,
    )


def pseudo_death_background_mask(
    diagnostics: Sequence[NoPriorContinuationDiagnostics],
    *,
    config: NoPriorContinuationConfig | None = None,
) -> np.ndarray:
    """Return weak label-free candidates as death/background examples."""

    cfg = config or NoPriorContinuationConfig()
    anchors = pseudo_continuation_anchor_mask(diagnostics, config=cfg)
    weak = np.asarray(
        [
            _finite(item.registered_iou) <= float(cfg.max_background_registered_iou)
            or _finite(item.shifted_iou) <= float(cfg.max_background_shifted_iou)
            or _finite(item.growth_mahalanobis) >= float(
                cfg.min_background_growth_mahalanobis
            )
            or _finite(item.growth_residual) >= float(
                cfg.min_background_growth_residual
            )
            or _finite(item.min_cell_probability) <= float(
                cfg.max_background_cell_probability
            )
            or _finite(item.local_deformation) >= float(
                cfg.min_background_local_deformation
            )
            for item in diagnostics
        ],
        dtype=bool,
    )
    return weak & ~anchors


def score_no_prior_continuation(
    diagnostics: NoPriorContinuationDiagnostics,
    model: NoPriorContinuationModel,
) -> float:
    """Score one no-prior continuation with a calibrated model."""

    return float(model.log_continuation_ratio((diagnostics,))[0])


def _feature_row(item: NoPriorContinuationDiagnostics) -> tuple[float, ...]:
    area_ratio = _finite(item.area_ratio, default=0.0)
    area_similarity = min(area_ratio, 1.0 / area_ratio) if area_ratio > 0.0 else 0.0
    return (
        _finite(item.registered_iou),
        _finite(item.shifted_iou),
        _finite(item.min_cell_probability),
        area_similarity,
        _finite(item.threshold_margin),
        -_finite(item.centroid_distance),
        -_finite(item.growth_residual),
        -_finite(item.growth_mahalanobis),
        -_finite(item.local_deformation),
        -math.log1p(max(0, int(item.row_rank) - 1)),
        -math.log1p(max(0, int(item.column_rank) - 1)),
    )


def _empty_model(config: NoPriorContinuationConfig) -> NoPriorContinuationModel:
    n_features = len(FEATURE_NAMES)
    return NoPriorContinuationModel(
        feature_names=FEATURE_NAMES,
        continuation_location=np.zeros(n_features, dtype=float),
        continuation_scale=np.ones(n_features, dtype=float),
        background_location=np.zeros(n_features, dtype=float),
        background_scale=np.ones(n_features, dtype=float),
        used_features=np.zeros(n_features, dtype=bool),
        per_feature_clip=float(config.per_feature_clip),
        score_clip=float(config.score_clip),
    )


def _finite(value: float, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _finite_column(values: np.ndarray) -> np.ndarray:
    column = np.asarray(values, dtype=float).reshape(-1)
    return column[np.isfinite(column)]


def _robust_location_scale(values: np.ndarray, *, min_scale: float) -> tuple[float, float]:
    finite = _finite_column(values)
    if finite.size == 0:
        return 0.0, max(float(min_scale), 1.0)
    location = float(np.median(finite))
    mad = float(np.median(np.abs(finite - location)))
    return location, max(float(min_scale), 1.4826 * mad)


def _gaussian_log_density(
    values: np.ndarray, *, location: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    safe_scale = np.maximum(np.asarray(scale, dtype=float), 1.0e-6)
    centered = np.asarray(values, dtype=float) - np.asarray(location, dtype=float)
    standardized = centered / safe_scale
    return -0.5 * standardized**2 - np.log(safe_scale)
