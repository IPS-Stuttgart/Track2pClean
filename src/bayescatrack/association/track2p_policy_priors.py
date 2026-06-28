"""Track2p-policy edge priors for independent global-assignment ablations."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

PolicyThresholdMethod = Literal["otsu", "min"]
SessionEdge = tuple[int, int]
AUTO_IOU_COMPONENTS = ("iou_for_cost", "shifted_iou", "iou")


def _integer_like(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")

    try:
        return int(operator.index(value))
    except TypeError:
        pass

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must be an integer")
        try:
            numeric_value = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    elif isinstance(value, (float, np.floating)):
        numeric_value = float(value)
    else:
        raise ValueError(f"{name} must be an integer")

    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric_value)


def _nonnegative_int(value: Any, *, name: str) -> int:
    integer_value = _integer_like(value, name=name)
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer_value


def _optional_positive_int(value: Any | None, *, name: str) -> int | None:
    if value is None:
        return None
    integer_value = _integer_like(value, name=name)
    if integer_value < 1:
        raise ValueError(f"{name} must be at least 1 when provided")
    return integer_value


def _finite_float(
    value: Any,
    *,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> float:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{name} must be a finite float")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite float, not boolean")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    if min_value is not None:
        if min_inclusive:
            invalid_min = numeric_value < min_value
        else:
            invalid_min = numeric_value <= min_value
        if invalid_min:
            comparator = "at least" if min_inclusive else "greater than"
            raise ValueError(f"{name} must be {comparator} {min_value}")
    if max_value is not None:
        if max_inclusive:
            invalid_max = numeric_value > max_value
        else:
            invalid_max = numeric_value >= max_value
        if invalid_max:
            comparator = "at most" if max_inclusive else "less than"
            raise ValueError(f"{name} must be {comparator} {max_value}")
    return numeric_value


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, np.bool_):
        return bool(value)
    if type(value) is bool:
        return value
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Track2pPolicyPriorConfig:
    """Add a Track2p-like edge prior without reading Track2p output tracks."""

    threshold_method: PolicyThresholdMethod = "min"
    iou_component: str = "auto"
    relief: float = 0.5
    accepted_cost_cap: float | None = None
    non_policy_penalty: float = 0.0
    min_cost: float = -2.0
    max_gap: int | None = None
    consecutive_only: bool = False
    row_top_k: int = 0
    column_top_k: int = 0
    mutual_top_k: int = 0
    rescue_min_iou: float = 0.0
    rescue_margin: float = 0.0
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        if self.threshold_method not in {"otsu", "min"}:
            raise ValueError("threshold_method must be 'otsu' or 'min'")
        if not str(self.iou_component).strip():
            raise ValueError("iou_component must be a non-empty string")
        object.__setattr__(
            self,
            "consecutive_only",
            _strict_bool(self.consecutive_only, name="consecutive_only"),
        )
        for name in ("relief", "non_policy_penalty", "rescue_margin"):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name, min_value=0.0),
            )
        if self.accepted_cost_cap is not None:
            object.__setattr__(
                self,
                "accepted_cost_cap",
                _finite_float(self.accepted_cost_cap, name="accepted_cost_cap"),
            )
        object.__setattr__(
            self,
            "min_cost",
            _finite_float(self.min_cost, name="min_cost"),
        )
        object.__setattr__(
            self,
            "max_gap",
            _optional_positive_int(self.max_gap, name="max_gap"),
        )
        object.__setattr__(
            self,
            "row_top_k",
            _nonnegative_int(self.row_top_k, name="row_top_k"),
        )
        object.__setattr__(
            self,
            "column_top_k",
            _nonnegative_int(self.column_top_k, name="column_top_k"),
        )
        object.__setattr__(
            self,
            "mutual_top_k",
            _nonnegative_int(self.mutual_top_k, name="mutual_top_k"),
        )
        object.__setattr__(
            self,
            "rescue_min_iou",
            _finite_float(
                self.rescue_min_iou,
                name="rescue_min_iou",
                min_value=0.0,
                max_value=1.0,
            ),
        )
        object.__setattr__(
            self,
            "large_cost",
            _finite_float(
                self.large_cost,
                name="large_cost",
                min_value=0.0,
                min_inclusive=False,
            ),
        )


def track2p_policy_prior_config_from_mapping(
    value: Track2pPolicyPriorConfig | Mapping[str, Any] | None,
) -> Track2pPolicyPriorConfig | None:
    """Normalize optional Track2p-policy-prior configuration values."""

    if value is None:
        return None
    if isinstance(value, Track2pPolicyPriorConfig):
        return value
    return Track2pPolicyPriorConfig(**dict(value))


def apply_track2p_policy_edge_prior(
    cost_matrix: np.ndarray,
    pairwise_components: Mapping[str, np.ndarray],
    *,
    session_gap: int,
    config: Track2pPolicyPriorConfig | Mapping[str, Any] | None,
) -> np.ndarray:
    """Return ``cost_matrix`` adjusted by an independent Track2p-policy prior."""

    cfg = track2p_policy_prior_config_from_mapping(config)
    if cfg is None:
        return np.asarray(cost_matrix, dtype=float).copy()
    gap = int(session_gap)
    if cfg.consecutive_only and gap != 1:
        return np.asarray(cost_matrix, dtype=float).copy()
    if cfg.max_gap is not None and gap > int(cfg.max_gap):
        return np.asarray(cost_matrix, dtype=float).copy()

    iou_matrix = _policy_iou_component(pairwise_components, cfg.iou_component)
    policy_mask = track2p_policy_edge_mask(iou_matrix, config=cfg)
    costs = np.asarray(cost_matrix, dtype=float).copy()
    if policy_mask.shape != costs.shape:
        raise ValueError(
            "Track2p-policy prior mask shape mismatch: "
            f"expected {costs.shape}, got {policy_mask.shape}"
        )

    finite = np.isfinite(costs) & (costs < float(cfg.large_cost))
    if cfg.non_policy_penalty > 0.0:
        costs[finite & ~policy_mask] += float(cfg.non_policy_penalty)
    policy_candidates = policy_mask & finite
    if not np.any(policy_candidates):
        return costs

    if cfg.accepted_cost_cap is not None:
        costs[policy_candidates] = np.minimum(
            costs[policy_candidates], float(cfg.accepted_cost_cap)
        )
    if cfg.relief > 0.0:
        costs[policy_candidates] -= float(cfg.relief)
    costs[policy_candidates] = np.maximum(costs[policy_candidates], float(cfg.min_cost))
    return costs


def track2p_policy_edge_mask(
    iou_matrix: np.ndarray,
    *,
    config: Track2pPolicyPriorConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return policy-supported edges from a registered IoU-like matrix."""

    cfg = track2p_policy_prior_config_from_mapping(config) or Track2pPolicyPriorConfig()
    iou = np.asarray(iou_matrix, dtype=float)
    if iou.ndim != 2:
        raise ValueError("iou_matrix must be two-dimensional")
    mask = np.zeros(iou.shape, dtype=bool)
    if iou.size == 0 or iou.shape[0] == 0 or iou.shape[1] == 0:
        return mask

    finite_iou = np.where(np.isfinite(iou), iou, 0.0)
    finite_iou = np.clip(finite_iou, 0.0, 1.0)
    row_ind, col_ind = linear_sum_assignment(1.0 - finite_iou)
    assigned_iou = finite_iou[row_ind, col_ind]
    threshold = _threshold_assigned_iou(assigned_iou, method=cfg.threshold_method)
    accepted = (assigned_iou >= threshold) & (assigned_iou > 0.0)
    if np.any(accepted):
        mask[row_ind[accepted], col_ind[accepted]] = True

    _add_local_rescue_edges(
        mask,
        finite_iou,
        threshold=threshold,
        row_top_k=cfg.row_top_k,
        column_top_k=cfg.column_top_k,
        mutual_top_k=cfg.mutual_top_k,
        rescue_min_iou=float(cfg.rescue_min_iou),
        rescue_margin=float(cfg.rescue_margin),
    )
    return mask


def _policy_iou_component(
    pairwise_components: Mapping[str, np.ndarray], iou_component: str
) -> np.ndarray:
    component_name = str(iou_component)
    if component_name == "auto":
        for candidate in AUTO_IOU_COMPONENTS:
            if candidate in pairwise_components:
                return np.asarray(pairwise_components[candidate], dtype=float)
        available = ", ".join(sorted(pairwise_components)) or "<none>"
        raise KeyError(
            "No IoU-like pairwise component is available for the Track2p-policy "
            f"prior. Tried {AUTO_IOU_COMPONENTS!r}; available: {available}."
        )
    if component_name not in pairwise_components:
        available = ", ".join(sorted(pairwise_components)) or "<none>"
        raise KeyError(
            f"Pairwise component {component_name!r} is not available for the "
            f"Track2p-policy prior. Available components: {available}."
        )
    return np.asarray(pairwise_components[component_name], dtype=float)


def _add_local_rescue_edges(
    mask: np.ndarray,
    iou: np.ndarray,
    *,
    threshold: float,
    row_top_k: int,
    column_top_k: int,
    mutual_top_k: int,
    rescue_min_iou: float,
    rescue_margin: float,
) -> None:
    if row_top_k <= 0 and column_top_k <= 0 and mutual_top_k <= 0:
        return
    if not np.isfinite(threshold):
        return
    floor = max(0.0, float(rescue_min_iou), float(threshold) - float(rescue_margin))
    if row_top_k > 0:
        _add_axis_top_k_rescue_edges(mask, iou, top_k=row_top_k, floor=floor, axis=1)
    if column_top_k > 0:
        _add_axis_top_k_rescue_edges(mask, iou, top_k=column_top_k, floor=floor, axis=0)
    if mutual_top_k > 0:
        _add_mutual_top_k_rescue_edges(mask, iou, top_k=mutual_top_k, floor=floor)


def _add_axis_top_k_rescue_edges(
    mask: np.ndarray,
    iou: np.ndarray,
    *,
    top_k: int,
    floor: float,
    axis: int,
) -> None:
    if axis == 1:
        for row_index, row_values in enumerate(iou):
            columns = _top_k_above_floor(row_values, top_k=top_k, floor=floor)
            mask[row_index, columns] = True
        return
    if axis == 0:
        for column_index, column_values in enumerate(iou.T):
            rows = _top_k_above_floor(column_values, top_k=top_k, floor=floor)
            mask[rows, column_index] = True
        return
    raise ValueError("axis must be 0 or 1")


def _add_mutual_top_k_rescue_edges(
    mask: np.ndarray,
    iou: np.ndarray,
    *,
    top_k: int,
    floor: float,
) -> None:
    """Add edges that are simultaneously row-top-k and column-top-k.

    Unlike the one-sided row/column rescue modes, this is a high-precision local
    evidence prior: an edge must be among the best finite IoU candidates for the
    source ROI and for the target ROI. This is useful for Track2p-style policies
    where correct links frequently sit just below a global threshold but remain
    locally reciprocal, while split/merge clutter often appears only one-sided.
    """

    row_supported = np.zeros(iou.shape, dtype=bool)
    column_supported = np.zeros(iou.shape, dtype=bool)
    _add_axis_top_k_rescue_edges(row_supported, iou, top_k=top_k, floor=floor, axis=1)
    _add_axis_top_k_rescue_edges(
        column_supported,
        iou,
        top_k=top_k,
        floor=floor,
        axis=0,
    )
    mask[row_supported & column_supported] = True


def _top_k_above_floor(values: np.ndarray, *, top_k: int, floor: float) -> np.ndarray:
    candidates = np.flatnonzero(
        np.isfinite(values) & (values > 0.0) & (values >= float(floor))
    )
    if candidates.size == 0:
        return candidates
    ordered = candidates[np.argsort(values[candidates])[::-1]]
    return ordered[:top_k]


def _threshold_assigned_iou(
    values: np.ndarray, *, method: PolicyThresholdMethod
) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    if np.allclose(values, values[0]):
        value = float(values[0])
        if value > 0.0:
            return float(np.nextafter(value, -np.inf))
        return value
    if method == "otsu":
        return _otsu_threshold(values)
    if method == "min":
        positive = values[values > 0.0]
        if positive.size < 3 or np.allclose(positive, positive[0]):
            return _otsu_threshold(values)
        minimum = _minimum_threshold(positive)
        return minimum if np.isfinite(minimum) else _otsu_threshold(values)
    raise ValueError(f"Unsupported threshold method: {method!r}")


def _otsu_threshold(values: np.ndarray) -> float:
    try:
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(values))
    except Exception:  # pragma: no cover - exercised only without scikit-image
        return _numpy_otsu_threshold(values)


def _minimum_threshold(values: np.ndarray) -> float:
    try:
        from skimage.filters import threshold_minimum

        return float(threshold_minimum(values))
    except Exception:
        return float("nan")


def _numpy_otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("inf")
    if np.allclose(values, values[0]):
        return float(values[0])
    bins = int(np.clip(values.size, 2, 256))
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_left = np.cumsum(hist).astype(float)
    weight_right = np.cumsum(hist[::-1]).astype(float)[::-1]
    mean_left = np.cumsum(hist * centers) / np.maximum(weight_left, 1.0)
    mean_right = (
        np.cumsum((hist * centers)[::-1]) / np.maximum(weight_right[::-1], 1.0)
    )[::-1]
    between = (
        weight_left[:-1] * weight_right[1:] * (mean_left[:-1] - mean_right[1:]) ** 2
    )
    if between.size == 0:
        return float(np.min(values))
    return float(centers[int(np.argmax(between))])


__all__ = (
    "Track2pPolicyPriorConfig",
    "apply_track2p_policy_edge_prior",
    "track2p_policy_edge_mask",
    "track2p_policy_prior_config_from_mapping",
)
