"""Label-free identity-history consistency scoring for FullMHT.

The scan-assignment runner already scores whether a proposed edge is plausible
inside the current session pair.  This module scores a different question: is the
candidate plausible for *this identity history*?  It compares the candidate edge
against the same track's previous accepted edge diagnostics and returns a risk
only when two label-free signals agree:

* overlap/cell evidence becomes unusually weak for that identity; and
* growth or local-deformation evidence becomes unusually large for that identity.

That joint-evidence shape is deliberate.  A single weak feature should not be
enough to break a complete identity, but a candidate that is simultaneously a low
overlap outlier and a growth/motion outlier is exactly the kind of continuation a
full MHT should keep as an alternative miss/gap hypothesis instead of accepting
locally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

HISTORY_FEATURES: tuple[str, ...] = (
    "registered_iou",
    "shifted_iou",
    "min_cell_probability",
    "growth_residual",
    "growth_mahalanobis",
    "local_deformation",
)


@dataclass(frozen=True)
class IdentityHistoryConsistencyConfig:
    """Controls for label-free per-track continuation risk."""

    weight: float = 0.0
    min_history_edges: int = 2
    min_feature_scale: float = 0.05
    joint_margin: float = 1.0
    score_clip: float = 8.0


def identity_history_consistency_risk(
    history_edges: Sequence[Mapping[str, Any]],
    candidate_edge: Mapping[str, Any],
    *,
    config: IdentityHistoryConsistencyConfig | None = None,
) -> float:
    """Return a weighted risk for a candidate edge against its own track history.

    Inputs are plain diagnostic mappings so the scorer can be used from the
    runner, tests, exposure audits, or offline ledgers without depending on
    manual-GT columns.  Missing or non-finite values simply remove that feature
    from the corresponding one-sided comparison.
    """

    cfg = config or IdentityHistoryConsistencyConfig()
    if float(cfg.weight) <= 0.0:
        return 0.0

    history = _feature_matrix(history_edges)
    if history.shape[0] < max(1, int(cfg.min_history_edges)):
        return 0.0
    candidate = _feature_vector(candidate_edge)

    low_overlap = max(
        _one_sided_history_deviation(
            history[:, 0], candidate[0], direction=-1.0, config=cfg
        ),
        0.5
        * _one_sided_history_deviation(
            history[:, 1], candidate[1], direction=-1.0, config=cfg
        ),
        0.5
        * _one_sided_history_deviation(
            history[:, 2], candidate[2], direction=-1.0, config=cfg
        ),
    )
    growth_motion = max(
        _one_sided_history_deviation(
            history[:, 3], candidate[3], direction=1.0, config=cfg
        ),
        _one_sided_history_deviation(
            history[:, 4], candidate[4], direction=1.0, config=cfg
        ),
        0.5
        * _one_sided_history_deviation(
            history[:, 5], candidate[5], direction=1.0, config=cfg
        ),
    )
    joint_risk = min(float(low_overlap), float(growth_motion))
    unweighted = max(0.0, joint_risk - float(cfg.joint_margin))
    clip = max(0.0, float(cfg.score_clip))
    if clip > 0.0:
        unweighted = min(float(clip), unweighted)
    return float(cfg.weight) * float(unweighted)


def _feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(HISTORY_FEATURES)), dtype=float)
    matrix = np.asarray([_feature_vector(row) for row in rows], dtype=float)
    if matrix.ndim != 2:
        return np.zeros((0, len(HISTORY_FEATURES)), dtype=float)
    finite_rows = np.any(np.isfinite(matrix), axis=1)
    return matrix[np.asarray(finite_rows, dtype=bool)]


def _feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([_finite_or_nan(row.get(name)) for name in HISTORY_FEATURES], dtype=float)


def _finite_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _one_sided_history_deviation(
    history_values: np.ndarray,
    value: float,
    *,
    direction: float,
    config: IdentityHistoryConsistencyConfig,
) -> float:
    values = np.asarray(history_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < max(1, int(config.min_history_edges)):
        return 0.0
    if not np.isfinite(float(value)):
        return 0.0
    location, scale = _robust_location_scale(
        values, min_scale=float(config.min_feature_scale)
    )
    signed_deviation = float(direction) * (float(value) - float(location))
    return float(max(0.0, signed_deviation / max(float(scale), 1.0e-9)))


def _robust_location_scale(values: np.ndarray, *, min_scale: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, max(float(min_scale), 1.0)
    location = float(np.median(finite))
    mad = float(np.median(np.abs(finite - location)))
    scale = max(float(min_scale), 1.4826 * mad)
    return location, scale
