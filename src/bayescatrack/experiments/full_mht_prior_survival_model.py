"""Label-free prior-edge survival model for FullMHT.

The current FullMHT prior-veto row uses a strict hand-gated hazard pocket.  This
module is the next method layer: it turns Track2p prior-edge diagnostics into a
calibrated survival log-likelihood ratio that can later replace the fixed pocket
inside the scan-assignment score.

No manual-GT labels are accepted or inspected here.  Calibration uses
high-confidence pseudo-survival anchors and risky prior-edge background examples
selected from label-free geometry, growth, endpoint confidence, rank, and
component-history diagnostics.
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
    "negative_growth_residual",
    "negative_growth_mahalanobis",
    "negative_local_deformation",
    "negative_log_row_rank",
    "negative_log_column_rank",
    "terminal_edge",
    "last_session_edge",
    "complete_component",
)


@dataclass(frozen=True)
class PriorEdgeSurvivalDiagnostics:
    """Label-free diagnostics for one Track2p proposal/prior edge."""

    registered_iou: float
    shifted_iou: float
    growth_residual: float
    growth_mahalanobis: float
    min_cell_probability: float
    area_ratio: float = 1.0
    local_deformation: float = 0.0
    row_rank: int = 1
    column_rank: int = 1
    terminal_edge: bool = False
    last_session_edge: bool = False
    complete_component: bool = False


@dataclass(frozen=True)
class PriorEdgeSurvivalConfig:
    """Pseudo-label and likelihood controls for prior-edge survival."""

    min_anchor_registered_iou: float = 0.75
    min_anchor_shifted_iou: float = 0.65
    max_anchor_growth_mahalanobis: float = 1.5
    max_anchor_growth_residual: float = 1.5
    min_anchor_cell_probability: float = 0.80
    max_anchor_rank: int = 1
    max_background_registered_iou: float = 0.50
    max_background_shifted_iou: float = 0.55
    min_background_growth_mahalanobis: float = 2.5
    min_background_growth_residual: float = 2.0
    max_background_cell_probability: float = 0.70
    min_examples_per_class: int = 2
    min_feature_scale: float = 0.05
    per_feature_clip: float = 4.0
    score_clip: float = 8.0


@dataclass(frozen=True)
class PriorEdgeSurvivalModel:
    """Robust Gaussian likelihood-ratio model for prior-edge survival."""

    feature_names: tuple[str, ...]
    survival_location: np.ndarray
    survival_scale: np.ndarray
    background_location: np.ndarray
    background_scale: np.ndarray
    used_features: np.ndarray
    per_feature_clip: float
    score_clip: float

    @property
    def enabled(self) -> bool:
        return bool(np.any(np.asarray(self.used_features, dtype=bool)))

    def log_survival_ratio(
        self, diagnostics: Sequence[PriorEdgeSurvivalDiagnostics]
    ) -> np.ndarray:
        """Return positive-for-survival log-likelihood ratios."""

        features = prior_edge_feature_matrix(diagnostics)
        if features.shape[1] != len(self.feature_names):
            raise ValueError("Feature dimension mismatch for prior survival model")
        if not self.enabled:
            return np.zeros(features.shape[0], dtype=float)
        survival_ll = _gaussian_log_density(
            features,
            location=np.asarray(self.survival_location, dtype=float),
            scale=np.asarray(self.survival_scale, dtype=float),
        )
        background_ll = _gaussian_log_density(
            features,
            location=np.asarray(self.background_location, dtype=float),
            scale=np.asarray(self.background_scale, dtype=float),
        )
        llr = survival_ll - background_ll
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


def calibrate_prior_edge_survival_model(
    diagnostics: Sequence[PriorEdgeSurvivalDiagnostics],
    *,
    config: PriorEdgeSurvivalConfig | None = None,
) -> PriorEdgeSurvivalModel:
    """Fit a label-free prior-edge survival likelihood-ratio model."""

    cfg = config or PriorEdgeSurvivalConfig()
    features = prior_edge_feature_matrix(diagnostics)
    survival_mask = pseudo_survival_anchor_mask(diagnostics, config=cfg)
    background_mask = pseudo_hazard_background_mask(diagnostics, config=cfg)
    n_features = len(FEATURE_NAMES)
    empty = _empty_model(cfg)
    if features.shape[0] == 0 or features.shape[1] != n_features:
        return empty
    if int(np.sum(survival_mask)) < int(cfg.min_examples_per_class):
        return empty
    if int(np.sum(background_mask)) < int(cfg.min_examples_per_class):
        return empty

    survival = features[np.asarray(survival_mask, dtype=bool)]
    background = features[np.asarray(background_mask, dtype=bool)]
    survival_location = np.zeros(n_features, dtype=float)
    survival_scale = np.ones(n_features, dtype=float)
    background_location = np.zeros(n_features, dtype=float)
    background_scale = np.ones(n_features, dtype=float)
    used = np.zeros(n_features, dtype=bool)
    for index in range(n_features):
        surv_values = _finite_column(survival[:, index])
        bg_values = _finite_column(background[:, index])
        if surv_values.size < int(cfg.min_examples_per_class):
            continue
        if bg_values.size < int(cfg.min_examples_per_class):
            continue
        survival_location[index], survival_scale[index] = _robust_location_scale(
            surv_values, min_scale=float(cfg.min_feature_scale)
        )
        background_location[index], background_scale[index] = _robust_location_scale(
            bg_values, min_scale=float(cfg.min_feature_scale)
        )
        if not math.isclose(
            float(survival_location[index]),
            float(background_location[index]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            used[index] = True
    if not np.any(used):
        return empty
    return PriorEdgeSurvivalModel(
        feature_names=FEATURE_NAMES,
        survival_location=survival_location,
        survival_scale=survival_scale,
        background_location=background_location,
        background_scale=background_scale,
        used_features=used,
        per_feature_clip=float(cfg.per_feature_clip),
        score_clip=float(cfg.score_clip),
    )


def prior_edge_feature_matrix(
    diagnostics: Sequence[PriorEdgeSurvivalDiagnostics],
) -> np.ndarray:
    """Convert prior-edge diagnostics to a numeric feature matrix."""

    rows = [_prior_edge_feature_row(item) for item in diagnostics]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    return np.asarray(rows, dtype=float).reshape(len(rows), len(FEATURE_NAMES))


def pseudo_survival_anchor_mask(
    diagnostics: Sequence[PriorEdgeSurvivalDiagnostics],
    *,
    config: PriorEdgeSurvivalConfig | None = None,
) -> np.ndarray:
    """Return high-confidence label-free pseudo-survival anchors."""

    cfg = config or PriorEdgeSurvivalConfig()
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
            and int(item.row_rank) <= int(cfg.max_anchor_rank)
            and int(item.column_rank) <= int(cfg.max_anchor_rank)
            for item in diagnostics
        ],
        dtype=bool,
    )


def pseudo_hazard_background_mask(
    diagnostics: Sequence[PriorEdgeSurvivalDiagnostics],
    *,
    config: PriorEdgeSurvivalConfig | None = None,
) -> np.ndarray:
    """Return risky label-free background examples for prior-edge survival."""

    cfg = config or PriorEdgeSurvivalConfig()
    anchors = pseudo_survival_anchor_mask(diagnostics, config=cfg)
    risky = np.asarray(
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
            for item in diagnostics
        ],
        dtype=bool,
    )
    return risky & ~anchors


def score_prior_edge_survival(
    diagnostics: PriorEdgeSurvivalDiagnostics,
    model: PriorEdgeSurvivalModel,
) -> float:
    """Score one prior edge with a calibrated survival model."""

    return float(model.log_survival_ratio((diagnostics,))[0])


def _prior_edge_feature_row(item: PriorEdgeSurvivalDiagnostics) -> tuple[float, ...]:
    area_ratio = _finite(item.area_ratio, default=0.0)
    area_similarity = min(area_ratio, 1.0 / area_ratio) if area_ratio > 0.0 else 0.0
    return (
        _finite(item.registered_iou),
        _finite(item.shifted_iou),
        _finite(item.min_cell_probability),
        area_similarity,
        -_finite(item.growth_residual),
        -_finite(item.growth_mahalanobis),
        -_finite(item.local_deformation),
        -math.log1p(max(0, int(item.row_rank) - 1)),
        -math.log1p(max(0, int(item.column_rank) - 1)),
        float(bool(item.terminal_edge)),
        float(bool(item.last_session_edge)),
        float(bool(item.complete_component)),
    )


def _empty_model(config: PriorEdgeSurvivalConfig) -> PriorEdgeSurvivalModel:
    n_features = len(FEATURE_NAMES)
    return PriorEdgeSurvivalModel(
        feature_names=FEATURE_NAMES,
        survival_location=np.zeros(n_features, dtype=float),
        survival_scale=np.ones(n_features, dtype=float),
        background_location=np.zeros(n_features, dtype=float),
        background_scale=np.ones(n_features, dtype=float),
        used_features=np.zeros(n_features, dtype=bool),
        per_feature_clip=float(config.per_feature_clip),
        score_clip=float(config.score_clip),
    )


def _finite(value: float, *, default: float = 0.0) -> float:
    number = float(value)
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
