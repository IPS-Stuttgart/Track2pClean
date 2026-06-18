"""Split/merge segmentation-event diagnostics for longitudinal ROI tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ._numeric_validation import finite_nonnegative_float as _finite_nonnegative_float
from ._numeric_validation import positive_integer as _positive_integer
from ._numeric_validation import probability as _probability


@dataclass(frozen=True)
class SegmentationEventConfig:
    """Thresholds for candidate split/merge events."""

    min_overlap_fraction: float = 0.25
    min_weighted_dice: float = 0.20
    max_area_ratio_cost: float = 1.25
    min_children: int = 2
    max_children: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_overlap_fraction",
            _probability(self.min_overlap_fraction, name="min_overlap_fraction"),
        )
        object.__setattr__(
            self,
            "min_weighted_dice",
            _probability(self.min_weighted_dice, name="min_weighted_dice"),
        )
        object.__setattr__(
            self,
            "max_area_ratio_cost",
            _finite_nonnegative_float(
                self.max_area_ratio_cost,
                name="max_area_ratio_cost",
            ),
        )
        min_children = _positive_integer(self.min_children, name="min_children")
        max_children = _positive_integer(self.max_children, name="max_children")
        if min_children < 2:
            raise ValueError("min_children must be at least two")
        if max_children < min_children:
            raise ValueError("max_children must be >= min_children")
        object.__setattr__(self, "min_children", min_children)
        object.__setattr__(self, "max_children", max_children)


@dataclass(frozen=True)
class SegmentationEventCandidate:
    """Potential one-to-many or many-to-one segmentation event."""

    event_type: str
    source_roi_index: int
    target_roi_indices: tuple[int, ...]
    score: float
    support_size: int


@dataclass(frozen=True)
class SegmentationEvent:
    """Compact split/merge event using matrix positions."""

    event_type: str
    reference_positions: tuple[int, ...]
    measurement_positions: tuple[int, ...]
    score: float


def detect_segmentation_events(
    pairwise_components: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | SegmentationEventConfig | None = None,
) -> list[SegmentationEvent]:
    """Detect simple split and merge candidates from pairwise similarity matrices."""

    similarity = _first_available(
        pairwise_components, ("weighted_dice_similarity", "iou")
    )
    if isinstance(config, SegmentationEventConfig):
        min_similarity = config.min_weighted_dice
        min_children = config.min_children
        max_children = config.max_children
    else:
        options = {} if config is None else dict(config)
        min_similarity = _probability(
            options.get("min_similarity", 0.20), name="min_similarity"
        )
        min_children = _positive_integer(
            options.get("min_children", 2), name="min_children"
        )
        max_children = _positive_integer(
            options.get("max_children", 4), name="max_children"
        )
        if min_children < 2:
            raise ValueError("min_children must be at least two")
        if max_children < min_children:
            raise ValueError("max_children must be >= min_children")
    events: list[SegmentationEvent] = []
    for row_index, row in enumerate(similarity):
        positions = np.flatnonzero(row >= min_similarity)
        if positions.size >= min_children:
            positions = positions[np.argsort(row[positions])[::-1]][:max_children]
            events.append(
                SegmentationEvent(
                    event_type="split",
                    reference_positions=(int(row_index),),
                    measurement_positions=tuple(int(pos) for pos in positions),
                    score=float(np.mean(row[positions])),
                )
            )
    for col_index in range(similarity.shape[1]):
        column = similarity[:, col_index]
        positions = np.flatnonzero(column >= min_similarity)
        if positions.size >= min_children:
            positions = positions[np.argsort(column[positions])[::-1]][:max_children]
            events.append(
                SegmentationEvent(
                    event_type="merge",
                    reference_positions=tuple(int(pos) for pos in positions),
                    measurement_positions=(int(col_index),),
                    score=float(np.mean(column[positions])),
                )
            )
    return events


def split_event_candidates(
    pairwise_components: Mapping[str, Any],
    *,
    reference_roi_indices: Sequence[int],
    measurement_roi_indices: Sequence[int],
    config: SegmentationEventConfig | None = None,
) -> list[SegmentationEventCandidate]:
    """Return candidate one-reference-to-many-measurement split events."""

    cfg, matrices, ref_indices, meas_indices = _candidate_inputs(
        pairwise_components,
        reference_roi_indices=reference_roi_indices,
        measurement_roi_indices=measurement_roi_indices,
        config=config,
    )

    candidates: list[SegmentationEventCandidate] = []
    for row_index, source_roi in enumerate(ref_indices):
        plausible = _plausible_child_mask(matrices, row_index, axis="row", config=cfg)
        child_positions = np.flatnonzero(plausible)
        if child_positions.size < cfg.min_children:
            continue
        child_positions = child_positions[
            np.argsort(matrices["score"][row_index, child_positions])[::-1]
        ][: cfg.max_children]
        score = float(np.mean(matrices["score"][row_index, child_positions]))
        candidates.append(
            SegmentationEventCandidate(
                event_type="split",
                source_roi_index=int(source_roi),
                target_roi_indices=tuple(
                    int(meas_indices[pos]) for pos in child_positions
                ),
                score=score,
                support_size=int(child_positions.size),
            )
        )
    return candidates


def merge_event_candidates(
    pairwise_components: Mapping[str, Any],
    *,
    reference_roi_indices: Sequence[int],
    measurement_roi_indices: Sequence[int],
    config: SegmentationEventConfig | None = None,
) -> list[SegmentationEventCandidate]:
    """Return candidate many-reference-to-one-measurement merge events."""

    cfg, matrices, ref_indices, meas_indices = _candidate_inputs(
        pairwise_components,
        reference_roi_indices=reference_roi_indices,
        measurement_roi_indices=measurement_roi_indices,
        config=config,
    )

    candidates: list[SegmentationEventCandidate] = []
    for col_index, target_roi in enumerate(meas_indices):
        plausible = _plausible_child_mask(
            matrices, col_index, axis="column", config=cfg
        )
        parent_positions = np.flatnonzero(plausible)
        if parent_positions.size < cfg.min_children:
            continue
        parent_positions = parent_positions[
            np.argsort(matrices["score"][parent_positions, col_index])[::-1]
        ][: cfg.max_children]
        score = float(np.mean(matrices["score"][parent_positions, col_index]))
        candidates.append(
            SegmentationEventCandidate(
                event_type="merge",
                source_roi_index=int(target_roi),
                target_roi_indices=tuple(
                    int(ref_indices[pos]) for pos in parent_positions
                ),
                score=score,
                support_size=int(parent_positions.size),
            )
        )
    return candidates


def segmentation_event_rows(
    candidates: Sequence[SegmentationEventCandidate],
) -> list[dict[str, int | float | str]]:
    """Serialize segmentation event candidates for reports."""

    return [
        {
            "event_type": candidate.event_type,
            "source_roi_index": candidate.source_roi_index,
            "target_roi_indices": ",".join(
                str(idx) for idx in candidate.target_roi_indices
            ),
            "score": candidate.score,
            "support_size": candidate.support_size,
        }
        for candidate in candidates
    ]


def event_soft_penalty_matrix(
    pairwise_components: Mapping[str, Any],
    *,
    config: SegmentationEventConfig | None = None,
) -> np.ndarray:
    """Return a penalty relief matrix for plausible split/merge edges.

    Values are negative or zero and can be added to a cost matrix to avoid
    over-penalizing edges involved in probable segmentation events.  The relief
    should be small; downstream one-to-one solvers still cannot fully represent
    one-to-many events, but this can prevent premature pruning of useful edges.
    """

    cfg = config or SegmentationEventConfig()
    matrices = _event_matrices(pairwise_components)
    score = matrices["score"]
    relief = np.zeros_like(score, dtype=float)
    plausible = (
        (matrices["overlap"] >= cfg.min_overlap_fraction)
        | (matrices["dice"] >= cfg.min_weighted_dice)
    ) & (matrices["area"] <= cfg.max_area_ratio_cost)
    relief[plausible] = -0.25 * np.clip(score[plausible], 0.0, 1.0)
    return relief


def _event_matrices(pairwise_components: Mapping[str, Any]) -> dict[str, np.ndarray]:
    overlap = _first_available(
        pairwise_components,
        (
            "overlap_min_fraction",
            "reference_containment",
            "measurement_containment",
            "iou",
        ),
    )
    dice = _first_available(pairwise_components, ("weighted_dice_similarity", "iou"))
    area = _first_available(pairwise_components, ("area_ratio_cost",))
    shape = overlap.shape
    if dice.shape != shape or area.shape != shape:
        raise ValueError("segmentation event components must have matching shapes")
    score = 0.5 * np.clip(overlap, 0.0, 1.0) + 0.5 * np.clip(dice, 0.0, 1.0)
    return {"overlap": overlap, "dice": dice, "area": area, "score": score}


def _candidate_inputs(
    pairwise_components: Mapping[str, Any],
    *,
    reference_roi_indices: Sequence[int],
    measurement_roi_indices: Sequence[int],
    config: SegmentationEventConfig | None,
) -> tuple[SegmentationEventConfig, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    cfg = config or SegmentationEventConfig()
    matrices = _event_matrices(pairwise_components)
    ref_indices = np.asarray(reference_roi_indices, dtype=int).reshape(-1)
    meas_indices = np.asarray(measurement_roi_indices, dtype=int).reshape(-1)
    _validate_shape(matrices["score"], ref_indices, meas_indices)
    return cfg, matrices, ref_indices, meas_indices


def _first_available(
    components: Mapping[str, Any], names: tuple[str, ...]
) -> np.ndarray:
    for name in names:
        if name in components:
            values = np.asarray(components[name], dtype=float)
            if values.ndim == 2:
                return np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=0.0)
    raise KeyError("None of the required components are available: " + ", ".join(names))


def _validate_shape(
    matrix: np.ndarray, ref_indices: np.ndarray, meas_indices: np.ndarray
) -> None:
    if matrix.shape != (ref_indices.size, meas_indices.size):
        raise ValueError("component shape does not match ROI index vectors")


def _plausible_child_mask(
    matrices: Mapping[str, np.ndarray],
    index: int,
    *,
    axis: str,
    config: SegmentationEventConfig,
) -> np.ndarray:
    if axis == "row":
        overlap = matrices["overlap"][index]
        dice = matrices["dice"][index]
        area = matrices["area"][index]
    elif axis == "column":
        overlap = matrices["overlap"][:, index]
        dice = matrices["dice"][:, index]
        area = matrices["area"][:, index]
    else:
        raise ValueError("axis must be 'row' or 'column'")
    return (
        (overlap >= config.min_overlap_fraction) | (dice >= config.min_weighted_dice)
    ) & (area <= config.max_area_ratio_cost)
