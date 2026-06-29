"""Strict validation for converting track hypotheses to matrices.

``hypotheses_to_matrix`` is an export boundary from beam-search candidates to a
dense row matrix.  The underlying implementation used a NumPy ``dtype=int``
conversion and filtered rows by the first hypothesis width, so malformed
hypotheses could either be silently dropped or have booleans/fractional values
coerced to ROI indices.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

from ._numeric_validation import integer as _integer

_PATCH_MARKER = "_bayescatrack_hypotheses_to_matrix_validation_patch"
_ROW_ERROR = "hypotheses must contain TrackHypothesis-like entries with row sequences"
_WIDTH_ERROR = "all hypothesis rows must have the same length"


def install_hypotheses_to_matrix_validation() -> None:
    """Install idempotent validation for hypothesis-matrix conversion."""

    from . import multi_hypothesis as module  # pylint: disable=import-outside-toplevel

    original = module.hypotheses_to_matrix
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def hypotheses_to_matrix_with_validation(hypotheses: Sequence[Any]) -> np.ndarray:
        rows = _normalize_hypothesis_rows(hypotheses)
        if not rows:
            return np.zeros((0, 0), dtype=int)
        return np.asarray(rows, dtype=int)

    setattr(hypotheses_to_matrix_with_validation, _PATCH_MARKER, True)
    setattr(hypotheses_to_matrix_with_validation, "_bayescatrack_original", original)
    module.hypotheses_to_matrix = hypotheses_to_matrix_with_validation


def _normalize_hypothesis_rows(
    hypotheses: Sequence[Any],
) -> tuple[tuple[int, ...], ...]:
    try:
        hypothesis_tuple = tuple(hypotheses)
    except TypeError as exc:
        raise ValueError(_ROW_ERROR) from exc
    if not hypothesis_tuple:
        return ()

    rows: list[tuple[int, ...]] = []
    expected_width: int | None = None
    for hypothesis_index, hypothesis in enumerate(hypothesis_tuple):
        row_name = f"hypotheses[{hypothesis_index}].row"
        raw_row = _hypothesis_row_values(hypothesis, name=row_name)
        if expected_width is None:
            expected_width = len(raw_row)
        elif len(raw_row) != expected_width:
            raise ValueError(_WIDTH_ERROR)
        rows.append(
            tuple(
                _integer(value, name=f"{row_name}[{value_index}]")
                for value_index, value in enumerate(raw_row)
            )
        )
    return tuple(rows)


def _hypothesis_row_values(hypothesis: Any, *, name: str) -> tuple[Any, ...]:
    try:
        row = hypothesis.row
    except AttributeError as exc:
        raise ValueError(_ROW_ERROR) from exc

    if isinstance(row, (str, bytes, bytearray, np.str_, np.bytes_)):
        raise ValueError(f"{name} must be a sequence of integer entries")
    try:
        row_values = tuple(row)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integer entries") from exc
    return row_values


__all__ = ["install_hypotheses_to_matrix_validation"]
