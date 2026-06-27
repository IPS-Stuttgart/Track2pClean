"""Track2p benchmark metric facade."""

# pylint: disable=wildcard-import,unused-wildcard-import

from __future__ import annotations

from typing import Any

import numpy as np
from bayescatrack.reference import Track2pReference

from . import complete_track_scores as _complete_track_scores
from .calibration_metrics import brier_score
from .complete_track_scores import *  # noqa: F401,F403
from .complete_track_scores import __all__ as _complete_track_score_exports
from .complete_track_scores import (
    _normalize_track_matrix_observations,
)
from .complete_track_scores import (
    normalize_track_matrix as _pyrecest_normalize_track_matrix,
)
from .track_error_ledger import *  # noqa: F401,F403
from .track_error_ledger import __all__ as _track_error_ledger_exports

__all__ = [
    *_complete_track_score_exports,
    *_track_error_ledger_exports,
    "brier_score",
    "score_track_matrix_against_reference",
]


def score_track_matrices(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Any | None = None,
    complete_session_indices: Any | None = None,
) -> dict[str, float | int]:
    """Delegate to the currently installed aggregate track scorer.

    The evaluation package installs validation wrappers on
    ``complete_track_scores.score_track_matrices`` during package initialization.
    Keep this facade dynamic so callers through this module use the wrapped
    scorer instead of an import-time function reference captured before wrapper
    installation.
    """

    return _complete_track_scores.score_track_matrices(
        predicted_track_matrix,
        reference_track_matrix,
        session_pairs=session_pairs,
        complete_session_indices=complete_session_indices,
    )


def normalize_track_matrix(track_matrix: Any) -> np.ndarray:
    """Normalize a track matrix after strict BayesCaTrack ROI-index validation."""

    normalized = _pyrecest_normalize_track_matrix(
        _normalize_track_matrix_observations(track_matrix, "track_matrix")
    )
    matrix = np.full(normalized.shape, -1, dtype=int)
    for index, value in np.ndenumerate(normalized):
        if value is not None:
            matrix[index] = int(value)
    return matrix


def score_track_matrix_against_reference(
    predicted_track_matrix: Any,
    reference: Track2pReference,
    *,
    curated_only: bool = False,
) -> dict[str, float | int]:
    """Score a predicted Suite2p-index track matrix against a Track2p reference."""

    curated_only = _normalize_curated_only_flag(curated_only)
    reference_matrix = normalize_track_matrix(reference.suite2p_indices)
    if curated_only:
        if reference.curated_mask is None:
            raise ValueError(
                "curated_only=True requires a Track2p reference with a curated_mask"
            )
        reference_matrix = reference_matrix[
            np.asarray(reference.curated_mask, dtype=bool)
        ]
    return score_track_matrices(predicted_track_matrix, reference_matrix)


def _normalize_curated_only_flag(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError("curated_only must be a boolean")
