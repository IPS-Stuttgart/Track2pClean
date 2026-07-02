"""Reference complete-track scoring normalizers.

The public in-memory scorer has enough reference context to interpret an empty
1-D prediction sequence as a zero-row matrix with one column per reference
session.  Without this normalization, ``[]`` is reshaped by the lower-level
nullable integer converter into shape ``(0, 1)`` and then rejected for every
multi-session reference.

The same wrapper also keeps seed-restricted complete-track scoring symmetric
when callers evaluate a subset of sessions that does not include the seed
session.  Rows outside the seed-session reference universe must not contribute
to the reference complete-track denominator.
"""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any

import numpy as np

from ._reference_scalar_validation import install_reference_scalar_validation

_PATCH_MARKER = "_bayescatrack_reference_empty_prediction_validation_patch"


def install_reference_empty_prediction_validation() -> None:
    """Install idempotent normalization for reference-based CT scorers."""

    from . import reference  # pylint: disable=import-outside-toplevel

    install_reference_scalar_validation(reference)

    original = reference.score_complete_tracks_against_reference
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def score_complete_tracks_against_reference_with_normalization(
        predicted_suite2p_indices: Any,
        reference_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        normalized_predictions = _normalize_empty_prediction_matrix(
            predicted_suite2p_indices,
            n_sessions=reference_obj.n_sessions,
        )
        return _score_against_seed_filtered_reference_if_needed(
            original,
            normalized_predictions,
            reference_obj,
            args=args,
            kwargs=kwargs,
        )

    setattr(
        score_complete_tracks_against_reference_with_normalization,
        _PATCH_MARKER,
        True,
    )
    setattr(
        score_complete_tracks_against_reference_with_normalization,
        "_bayescatrack_original",
        original,
    )
    reference.score_complete_tracks_against_reference = (
        score_complete_tracks_against_reference_with_normalization
    )


def _score_against_seed_filtered_reference_if_needed(
    original: Any,
    predicted_suite2p_indices: Any,
    reference_obj: Any,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    bound = signature(original).bind(
        predicted_suite2p_indices,
        reference_obj,
        *args,
        **kwargs,
    )
    bound.apply_defaults()
    if not bound.arguments.get("restrict_to_reference_seed_rois", True):
        return original(predicted_suite2p_indices, reference_obj, *args, **kwargs)

    from . import reference as reference_module  # pylint: disable=import-outside-toplevel

    curated_only = bound.arguments.get("curated_only", False)
    seed_session = reference_module._validate_session_index(  # pylint: disable=protected-access
        bound.arguments.get("seed_session", 0),
        reference_obj.n_sessions,
    )
    reference_indices = reference_obj.filtered_indices(curated_only=curated_only)
    seed_filtered_reference_indices = reference_module._filter_tracks_by_seed_rois(  # pylint: disable=protected-access
        reference_indices,
        reference_indices,
        seed_session=seed_session,
    )
    if seed_filtered_reference_indices.shape[0] == reference_indices.shape[0]:
        return original(predicted_suite2p_indices, reference_obj, *args, **kwargs)

    seed_filtered_reference = reference_module.Track2pReference(
        session_names=reference_obj.session_names,
        suite2p_indices=seed_filtered_reference_indices,
        session_dates=reference_obj.session_dates,
        curated_mask=np.ones((seed_filtered_reference_indices.shape[0],), dtype=bool),
        source=reference_obj.source,
    )
    return original(
        predicted_suite2p_indices,
        seed_filtered_reference,
        session_indices=bound.arguments.get("session_indices"),
        curated_only=False,
        seed_session=seed_session,
        restrict_to_reference_seed_rois=True,
    )


def _normalize_empty_prediction_matrix(value: Any, *, n_sessions: int) -> Any:
    array = np.asarray(value, dtype=object)
    if array.ndim == 1 and array.size == 0:
        return np.zeros((0, int(n_sessions)), dtype=object)
    return value


__all__ = ["install_reference_empty_prediction_validation"]
