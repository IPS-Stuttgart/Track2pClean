"""Strict ROI and label validation for edge-ranking diagnostics.

The edge-ranking helpers report ranks for original Suite2p ROI identifiers.  The
implementation used NumPy/Python integer coercion at the reporting boundary, so
malformed identifiers such as booleans, numeric strings, or fractional floats
could be silently converted into fabricated ROI IDs.  Label matrices likewise
must be validated before boolean casting so malformed GT entries cannot become
positive edges.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_edge_ranking_roi_validation_patch"
_ROI_ERROR_SUFFIX = "must contain non-negative integer ROI identifiers"
_LABEL_ERROR = "labels must be a binary matrix containing only 0/1 or boolean values"


def install_edge_ranking_roi_validation() -> None:
    """Install idempotent strict ROI and label validation on edge-ranking helpers."""

    from . import edge_ranking as _edge_ranking  # pylint: disable=import-outside-toplevel

    original_rank: Callable[..., Any] = _edge_ranking.rank_labeled_edges
    if not getattr(original_rank, _PATCH_MARKER, False):

        @wraps(original_rank)
        def rank_labeled_edges_with_roi_validation(
            labels: Any,
            score_matrices: Any,
            *,
            reference_roi_indices: Any,
            measurement_roi_indices: Any,
            score_directions: Any = None,
            metadata: Any = None,
        ) -> Any:
            return original_rank(
                _normalize_label_matrix(labels),
                score_matrices,
                reference_roi_indices=_normalize_roi_index_array(
                    reference_roi_indices,
                    "reference_roi_indices",
                    require_unique=True,
                ),
                measurement_roi_indices=_normalize_roi_index_array(
                    measurement_roi_indices,
                    "measurement_roi_indices",
                    require_unique=True,
                ),
                score_directions=score_directions,
                metadata=metadata,
            )

        setattr(rank_labeled_edges_with_roi_validation, _PATCH_MARKER, True)
        setattr(rank_labeled_edges_with_roi_validation, "_bayescatrack_original", original_rank)
        _edge_ranking.rank_labeled_edges = rank_labeled_edges_with_roi_validation

    original_missing: Callable[..., Any] = _edge_ranking.missing_reference_edge_rows
    if not getattr(original_missing, _PATCH_MARKER, False):

        @wraps(original_missing)
        def missing_reference_edge_rows_with_roi_validation(
            reference_matches: Any,
            *,
            reference_roi_indices: Any,
            measurement_roi_indices: Any,
            score_names: Any,
            score_directions: Any = None,
            metadata: Any = None,
        ) -> Any:
            return original_missing(
                _normalize_reference_matches(reference_matches),
                reference_roi_indices=_normalize_roi_index_array(
                    reference_roi_indices,
                    "reference_roi_indices",
                    require_unique=True,
                ),
                measurement_roi_indices=_normalize_roi_index_array(
                    measurement_roi_indices,
                    "measurement_roi_indices",
                    require_unique=True,
                ),
                score_names=score_names,
                score_directions=score_directions,
                metadata=metadata,
            )

        setattr(missing_reference_edge_rows_with_roi_validation, _PATCH_MARKER, True)
        setattr(missing_reference_edge_rows_with_roi_validation, "_bayescatrack_original", original_missing)
        _edge_ranking.missing_reference_edge_rows = missing_reference_edge_rows_with_roi_validation


def _normalize_label_matrix(labels: Any) -> np.ndarray:
    label_array = np.asarray(labels, dtype=object)
    if label_array.ndim != 2:
        raise ValueError("labels must be a two-dimensional matrix")
    normalized = np.zeros(label_array.shape, dtype=bool)
    for index, value in np.ndenumerate(label_array):
        normalized[index] = _normalize_label_value(value)
    return normalized


def _normalize_label_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        raise ValueError(_LABEL_ERROR)
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or numeric_value not in {0.0, 1.0}:
            raise ValueError(_LABEL_ERROR)
        return bool(int(numeric_value))
    try:
        integer_value = operator.index(value)
    except TypeError as exc:
        raise ValueError(_LABEL_ERROR) from exc
    integer_value = int(integer_value)
    if integer_value not in {0, 1}:
        raise ValueError(_LABEL_ERROR)
    return bool(integer_value)


def _normalize_reference_matches(reference_matches: Any) -> tuple[tuple[int, int], ...]:
    normalized_matches: list[tuple[int, int]] = []
    for match in reference_matches:
        try:
            reference_roi_index, measurement_roi_index = match
        except (TypeError, ValueError) as exc:
            raise ValueError("reference_matches must contain ROI-index pairs") from exc
        normalized_matches.append(
            (
                _normalize_roi_index(reference_roi_index, "reference_matches"),
                _normalize_roi_index(measurement_roi_index, "reference_matches"),
            )
        )
    return tuple(normalized_matches)


def _normalize_roi_index_array(
    values: Any,
    field_name: str,
    *,
    require_unique: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")

    normalized = [_normalize_roi_index(value, field_name) for value in array.tolist()]
    if require_unique and len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique ROI identifiers")
    return np.asarray(normalized, dtype=int)


def _normalize_roi_index(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} {_ROI_ERROR_SUFFIX}")

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} {_ROI_ERROR_SUFFIX}")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{field_name} {_ROI_ERROR_SUFFIX}") from exc

    integer_value = int(integer_value)
    if integer_value < 0:
        raise ValueError(f"{field_name} {_ROI_ERROR_SUFFIX}")
    return integer_value


__all__ = ["install_edge_ranking_roi_validation"]
