"""Vector-input normalization hooks for aggregate track-matrix scoring."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import ModuleType
from typing import Any

import numpy as np

_PATCH_ATTR = "_bayescatrack_track_matrix_vector_input_patch"


def install_track_matrix_vector_input_validation(scores_module: ModuleType) -> None:
    """Treat one-dimensional score_track_matrices inputs as single track rows."""

    original_score_track_matrices = scores_module.score_track_matrices
    if getattr(original_score_track_matrices, _PATCH_ATTR, False):
        return

    def _score_track_matrices_with_vector_rows(
        predicted_track_matrix: Any,
        reference_track_matrix: Any,
        *,
        session_pairs: Iterable[tuple[int, int]] | None = None,
        complete_session_indices: Sequence[int] | None = None,
    ) -> dict[str, float | int]:
        return original_score_track_matrices(
            _normalize_single_track_vector(predicted_track_matrix),
            _normalize_single_track_vector(reference_track_matrix),
            session_pairs=session_pairs,
            complete_session_indices=complete_session_indices,
        )

    setattr(_score_track_matrices_with_vector_rows, _PATCH_ATTR, True)
    setattr(
        _score_track_matrices_with_vector_rows,
        "_bayescatrack_original",
        original_score_track_matrices,
    )
    scores_module.score_track_matrices = _score_track_matrices_with_vector_rows


def _normalize_single_track_vector(track_matrix: Any) -> Any:
    array = np.asarray(track_matrix, dtype=object)
    if array.ndim != 1:
        return track_matrix
    if array.size == 0:
        return np.empty((0, 0), dtype=object)
    return array.reshape(1, -1)
