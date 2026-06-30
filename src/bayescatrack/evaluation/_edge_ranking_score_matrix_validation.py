"""Score-matrix name validation for edge-ranking diagnostics.

``rank_labeled_edges`` infers score direction from score names.  Without
normalizing score-matrix keys first, bytes keys such as ``b"iou"`` are converted
with ``str(...)`` to ``"b'iou'"`` and are therefore treated as costs instead of
similarities.  Distinct keys can also collapse to the same string name and
silently overwrite an earlier matrix.  Validate and normalize those names before
ranking.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_edge_ranking_score_matrix_validation_patch"
_SCORE_MATRIX_ERROR = "score_matrices must be a mapping with unique, non-empty string keys"
_SCORE_DIRECTION_ERROR = "score_directions must map unique score names to 'cost' or 'similarity'"


def install_edge_ranking_score_matrix_validation() -> None:
    """Install idempotent score-matrix name validation on edge-ranking helpers."""

    from . import (
        edge_ranking as _edge_ranking,  # pylint: disable=import-outside-toplevel
    )

    original: Callable[..., Any] = _edge_ranking.rank_labeled_edges
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def rank_labeled_edges_with_score_matrix_validation(
        labels: Any,
        score_matrices: Any,
        *,
        reference_roi_indices: Any,
        measurement_roi_indices: Any,
        score_directions: Any = None,
        metadata: Any = None,
    ) -> Any:
        return original(
            labels,
            _normalize_score_matrices(score_matrices),
            reference_roi_indices=reference_roi_indices,
            measurement_roi_indices=measurement_roi_indices,
            score_directions=_normalize_score_directions(score_directions),
            metadata=metadata,
        )

    setattr(rank_labeled_edges_with_score_matrix_validation, _PATCH_MARKER, True)
    setattr(
        rank_labeled_edges_with_score_matrix_validation,
        "_bayescatrack_original",
        original,
    )
    _edge_ranking.rank_labeled_edges = rank_labeled_edges_with_score_matrix_validation


def _normalize_score_matrices(score_matrices: Any) -> dict[str, Any]:
    if not isinstance(score_matrices, Mapping):
        raise ValueError(_SCORE_MATRIX_ERROR)

    normalized: dict[str, Any] = {}
    for score_name, score_values in score_matrices.items():
        normalized_name = _normalize_score_name(score_name, error_message=_SCORE_MATRIX_ERROR)
        if normalized_name in normalized:
            raise ValueError(_SCORE_MATRIX_ERROR)
        normalized[normalized_name] = score_values
    return normalized


def _normalize_score_directions(score_directions: Any) -> dict[str, Any] | None:
    if score_directions is None:
        return None
    if not isinstance(score_directions, Mapping):
        raise ValueError(_SCORE_DIRECTION_ERROR)

    normalized: dict[str, Any] = {}
    for score_name, direction in score_directions.items():
        normalized_name = _normalize_score_name(
            score_name,
            error_message=_SCORE_DIRECTION_ERROR,
        )
        if normalized_name in normalized:
            raise ValueError(_SCORE_DIRECTION_ERROR)
        if direction not in {"cost", "similarity"}:
            raise ValueError(_SCORE_DIRECTION_ERROR)
        normalized[normalized_name] = direction
    return normalized


def _normalize_score_name(score_name: Any, *, error_message: str) -> str:
    if isinstance(score_name, bytes):
        try:
            score_name = score_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(error_message) from exc
    if not isinstance(score_name, (str, np.str_)):
        raise ValueError(error_message)
    normalized = str(score_name)
    if not normalized:
        raise ValueError(error_message)
    return normalized


__all__ = ["install_edge_ranking_score_matrix_validation"]
