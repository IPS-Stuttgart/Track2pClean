"""Local competition features for pairwise ROI association.

These components turn a raw pairwise score matrix into per-candidate rank,
margin, ratio, mutual-best, and ambiguity features.  They are deliberately
computed from already available pairwise components, so they can be used both by
direct hand-weighted association costs and by calibrated/ranking models trained
from Track2p/manual ground truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

LOCAL_COMPETITION_ASSOCIATION_FEATURES: tuple[str, ...] = (
    "iou_row_rank_fraction",
    "iou_column_rank_fraction",
    "iou_competition_rank_cost",
    "iou_row_gap_to_best",
    "iou_column_gap_to_best",
    "iou_row_top_margin",
    "iou_column_top_margin",
    "iou_row_ratio_to_best",
    "iou_column_ratio_to_best",
    "iou_mutual_top1",
    "iou_row_candidate_fraction",
    "iou_column_candidate_fraction",
    "centroid_row_rank_fraction",
    "centroid_column_rank_fraction",
    "centroid_competition_rank_cost",
    "centroid_row_gap_to_best",
    "centroid_column_gap_to_best",
    "centroid_mutual_top1",
)


def add_local_competition_components(
    pairwise_components: Mapping[str, Any],
    *,
    similarity_component: str = "iou",
    distance_component: str = "centroid_distance",
    candidate_similarity_threshold: float = 0.0,
    epsilon: float = 1.0e-12,
) -> dict[str, np.ndarray]:
    """Return pairwise components augmented with local competition terms.

    Parameters
    ----------
    pairwise_components
        Mapping of existing pairwise component matrices.  If ``iou`` is present,
        score-competition features are added.  If ``centroid_distance`` is
        present, distance-rank competition features are added.
    similarity_component
        Name of a component where larger values are better, usually ``iou``.
    distance_component
        Name of a component where smaller values are better, usually
        ``centroid_distance``.
    candidate_similarity_threshold
        Threshold used to count plausible row/column competitors for the
        similarity component.
    epsilon
        Positive numerical stabilizer for ratios.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive")

    components = {key: np.asarray(value) for key, value in pairwise_components.items()}
    if similarity_component in components:
        components.update(
            pairwise_similarity_competition_components(
                components[similarity_component],
                prefix=_component_prefix(similarity_component),
                candidate_threshold=candidate_similarity_threshold,
                epsilon=epsilon,
            )
        )
    if distance_component in components:
        components.update(
            pairwise_distance_competition_components(
                components[distance_component],
                prefix=_component_prefix(distance_component),
            )
        )
    return components


def pairwise_similarity_competition_components(
    scores: Any,
    *,
    prefix: str,
    candidate_threshold: float = 0.0,
    epsilon: float = 1.0e-12,
) -> dict[str, np.ndarray]:
    """Return competition features for a matrix where larger scores are better."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive")
    score_matrix = _finite_pairwise_matrix(scores, fill_value=0.0)
    n_rows, n_columns = score_matrix.shape
    if n_rows == 0 or n_columns == 0:
        empty = np.zeros((n_rows, n_columns), dtype=float)
        return _empty_similarity_components(prefix, empty)

    row_rank = _rank_descending_by_row(score_matrix)
    column_rank = _rank_descending_by_column(score_matrix)
    row_rank_fraction = row_rank / max(n_columns - 1, 1)
    column_rank_fraction = column_rank / max(n_rows - 1, 1)

    row_best, row_second = _row_best_and_second(score_matrix)
    column_best, column_second = _column_best_and_second(score_matrix)
    row_best_matrix = row_best[:, None]
    column_best_matrix = column_best[None, :]
    row_second_matrix = row_second[:, None]
    column_second_matrix = column_second[None, :]

    row_top = row_rank == 0.0
    column_top = column_rank == 0.0
    row_alternative = np.where(row_top, row_second_matrix, row_best_matrix)
    column_alternative = np.where(column_top, column_second_matrix, column_best_matrix)

    row_ratio = np.zeros_like(score_matrix, dtype=float)
    column_ratio = np.zeros_like(score_matrix, dtype=float)
    np.divide(
        score_matrix,
        row_best_matrix,
        out=row_ratio,
        where=row_best_matrix > epsilon,
    )
    np.divide(
        score_matrix,
        column_best_matrix,
        out=column_ratio,
        where=column_best_matrix > epsilon,
    )

    candidate_threshold = float(candidate_threshold)
    candidate_mask = score_matrix > candidate_threshold
    row_candidate_count = np.count_nonzero(candidate_mask, axis=1).astype(float)
    column_candidate_count = np.count_nonzero(candidate_mask, axis=0).astype(float)

    return {
        f"{prefix}_row_rank": row_rank,
        f"{prefix}_column_rank": column_rank,
        f"{prefix}_row_rank_fraction": row_rank_fraction,
        f"{prefix}_column_rank_fraction": column_rank_fraction,
        f"{prefix}_competition_rank_cost": 0.5
        * (row_rank_fraction + column_rank_fraction),
        f"{prefix}_row_best": np.broadcast_to(row_best_matrix, score_matrix.shape),
        f"{prefix}_column_best": np.broadcast_to(
            column_best_matrix, score_matrix.shape
        ),
        f"{prefix}_row_second_best": np.broadcast_to(
            row_second_matrix, score_matrix.shape
        ),
        f"{prefix}_column_second_best": np.broadcast_to(
            column_second_matrix, score_matrix.shape
        ),
        f"{prefix}_row_gap_to_best": np.maximum(row_best_matrix - score_matrix, 0.0),
        f"{prefix}_column_gap_to_best": np.maximum(
            column_best_matrix - score_matrix, 0.0
        ),
        f"{prefix}_row_top_margin": score_matrix - row_alternative,
        f"{prefix}_column_top_margin": score_matrix - column_alternative,
        f"{prefix}_row_ratio_to_best": np.clip(row_ratio, 0.0, 1.0),
        f"{prefix}_column_ratio_to_best": np.clip(column_ratio, 0.0, 1.0),
        f"{prefix}_mutual_top1": (
            row_top
            & column_top
            & (row_best_matrix > candidate_threshold)
            & (column_best_matrix > candidate_threshold)
        ).astype(float),
        f"{prefix}_row_candidate_count": np.broadcast_to(
            row_candidate_count[:, None], score_matrix.shape
        ),
        f"{prefix}_column_candidate_count": np.broadcast_to(
            column_candidate_count[None, :], score_matrix.shape
        ),
        f"{prefix}_row_candidate_fraction": np.broadcast_to(
            row_candidate_count[:, None] / max(n_columns, 1), score_matrix.shape
        ),
        f"{prefix}_column_candidate_fraction": np.broadcast_to(
            column_candidate_count[None, :] / max(n_rows, 1), score_matrix.shape
        ),
    }


def pairwise_distance_competition_components(
    distances: Any,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Return competition features for a matrix where smaller values are better."""

    distance_matrix = _finite_pairwise_matrix(distances, fill_value=np.inf)
    n_rows, n_columns = distance_matrix.shape
    if n_rows == 0 or n_columns == 0:
        empty = np.zeros((n_rows, n_columns), dtype=float)
        return _empty_distance_components(prefix, empty)

    row_rank = _rank_ascending_by_row(distance_matrix)
    column_rank = _rank_ascending_by_column(distance_matrix)
    row_rank_fraction = row_rank / max(n_columns - 1, 1)
    column_rank_fraction = column_rank / max(n_rows - 1, 1)
    row_best = np.min(distance_matrix, axis=1)
    column_best = np.min(distance_matrix, axis=0)

    return {
        f"{prefix}_row_rank": row_rank,
        f"{prefix}_column_rank": column_rank,
        f"{prefix}_row_rank_fraction": row_rank_fraction,
        f"{prefix}_column_rank_fraction": column_rank_fraction,
        f"{prefix}_competition_rank_cost": 0.5
        * (row_rank_fraction + column_rank_fraction),
        f"{prefix}_row_gap_to_best": np.maximum(
            distance_matrix - row_best[:, None], 0.0
        ),
        f"{prefix}_column_gap_to_best": np.maximum(
            distance_matrix - column_best[None, :], 0.0
        ),
        f"{prefix}_mutual_top1": ((row_rank == 0.0) & (column_rank == 0.0)).astype(
            float
        ),
    }


def _component_prefix(component_name: str) -> str:
    if component_name == "centroid_distance":
        return "centroid"
    if component_name.endswith("_similarity"):
        return component_name[: -len("_similarity")]
    if component_name.endswith("_distance"):
        return component_name[: -len("_distance")]
    return component_name


def _finite_pairwise_matrix(values: Any, *, fill_value: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("pairwise competition components must be two-dimensional")
    return np.nan_to_num(
        matrix,
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )


def _rank_descending_by_row(values: np.ndarray) -> np.ndarray:
    n_rows, n_columns = values.shape
    ranks = np.zeros((n_rows, n_columns), dtype=float)
    order = np.argsort(-values, axis=1, kind="stable")
    ranks[np.arange(n_rows)[:, None], order] = np.arange(n_columns)[None, :]
    return ranks


def _rank_descending_by_column(values: np.ndarray) -> np.ndarray:
    n_rows, n_columns = values.shape
    ranks = np.zeros((n_rows, n_columns), dtype=float)
    order = np.argsort(-values, axis=0, kind="stable")
    ranks[order, np.arange(n_columns)[None, :]] = np.arange(n_rows)[:, None]
    return ranks


def _rank_ascending_by_row(values: np.ndarray) -> np.ndarray:
    n_rows, n_columns = values.shape
    ranks = np.zeros((n_rows, n_columns), dtype=float)
    order = np.argsort(values, axis=1, kind="stable")
    ranks[np.arange(n_rows)[:, None], order] = np.arange(n_columns)[None, :]
    return ranks


def _rank_ascending_by_column(values: np.ndarray) -> np.ndarray:
    n_rows, n_columns = values.shape
    ranks = np.zeros((n_rows, n_columns), dtype=float)
    order = np.argsort(values, axis=0, kind="stable")
    ranks[order, np.arange(n_columns)[None, :]] = np.arange(n_rows)[:, None]
    return ranks


def _row_best_and_second(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values, axis=1)[:, ::-1]
    best = sorted_values[:, 0]
    second = sorted_values[:, 1] if values.shape[1] > 1 else best.copy()
    return best, second


def _column_best_and_second(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values, axis=0)[::-1, :]
    best = sorted_values[0, :]
    second = sorted_values[1, :] if values.shape[0] > 1 else best.copy()
    return best, second


def _empty_similarity_components(prefix: str, empty: np.ndarray) -> dict[str, np.ndarray]:
    names = (
        "row_rank",
        "column_rank",
        "row_rank_fraction",
        "column_rank_fraction",
        "competition_rank_cost",
        "row_best",
        "column_best",
        "row_second_best",
        "column_second_best",
        "row_gap_to_best",
        "column_gap_to_best",
        "row_top_margin",
        "column_top_margin",
        "row_ratio_to_best",
        "column_ratio_to_best",
        "mutual_top1",
        "row_candidate_count",
        "column_candidate_count",
        "row_candidate_fraction",
        "column_candidate_fraction",
    )
    return {f"{prefix}_{name}": empty.copy() for name in names}


def _empty_distance_components(prefix: str, empty: np.ndarray) -> dict[str, np.ndarray]:
    names = (
        "row_rank",
        "column_rank",
        "row_rank_fraction",
        "column_rank_fraction",
        "competition_rank_cost",
        "row_gap_to_best",
        "column_gap_to_best",
        "mutual_top1",
    )
    return {f"{prefix}_{name}": empty.copy() for name in names}
