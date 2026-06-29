"""Empty-prediction normalization for reference-based complete-track scoring.

The public in-memory scorer has enough reference context to interpret an empty
1-D prediction sequence as a zero-row matrix with one column per reference
session.  Without this normalization, ``[]`` is reshaped by the lower-level
nullable integer converter into shape ``(0, 1)`` and then rejected for every
multi-session reference.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from ._reference_scalar_validation import install_reference_scalar_validation

_PATCH_MARKER = "_bayescatrack_reference_empty_prediction_validation_patch"


def install_reference_empty_prediction_validation() -> None:
    """Install idempotent empty-prediction handling for reference scorers."""

    from . import reference  # pylint: disable=import-outside-toplevel

    install_reference_scalar_validation(reference)

    original = reference.score_complete_tracks_against_reference
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def score_complete_tracks_against_reference_with_empty_predictions(
        predicted_suite2p_indices: Any,
        reference_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        normalized_predictions = _normalize_empty_prediction_matrix(
            predicted_suite2p_indices,
            n_sessions=reference_obj.n_sessions,
        )
        return original(normalized_predictions, reference_obj, *args, **kwargs)

    setattr(
        score_complete_tracks_against_reference_with_empty_predictions,
        _PATCH_MARKER,
        True,
    )
    setattr(
        score_complete_tracks_against_reference_with_empty_predictions,
        "_bayescatrack_original",
        original,
    )
    reference.score_complete_tracks_against_reference = (
        score_complete_tracks_against_reference_with_empty_predictions
    )


def _normalize_empty_prediction_matrix(value: Any, *, n_sessions: int) -> Any:
    array = np.asarray(value, dtype=object)
    if array.ndim == 1 and array.size == 0:
        return np.zeros((0, int(n_sessions)), dtype=object)
    return value


__all__ = ["install_reference_empty_prediction_validation"]
