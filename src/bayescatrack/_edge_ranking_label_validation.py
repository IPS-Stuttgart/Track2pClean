"""Validate edge-ranking label matrices."""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCH_MARKER = "_track2pclean_edge_label_validation_installed"
_ERROR_MESSAGE = "labels must contain only 0/1 values"


def install_edge_ranking_label_validation(edge_ranking_module: Any) -> None:
    """Install idempotent 0/1-label validation for edge-ranking diagnostics."""

    original_as_label_matrix = edge_ranking_module._as_label_matrix
    if getattr(original_as_label_matrix, _PATCH_MARKER, False):
        return

    def _as_label_matrix_with_label_validation(labels: Any) -> np.ndarray:
        label_matrix = np.asarray(labels, dtype=object)
        if label_matrix.ndim != 2:
            raise ValueError("labels must be a two-dimensional matrix")

        output = np.empty(label_matrix.shape, dtype=bool)
        for index, value in np.ndenumerate(label_matrix):
            output[index] = _parse_label(value)
        return output

    setattr(_as_label_matrix_with_label_validation, _PATCH_MARKER, True)
    setattr(
        _as_label_matrix_with_label_validation,
        "_bayescatrack_original",
        original_as_label_matrix,
    )
    edge_ranking_module._as_label_matrix = _as_label_matrix_with_label_validation


def _parse_label(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(_ERROR_MESSAGE)
        value = value.item()

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        raise ValueError(_ERROR_MESSAGE)

    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc
    if not np.isfinite(numeric_value) or numeric_value not in (0.0, 1.0):
        raise ValueError(_ERROR_MESSAGE)
    return bool(numeric_value)


__all__ = ["install_edge_ranking_label_validation"]
