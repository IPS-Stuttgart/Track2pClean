"""Input-shape validation for post-solve relinking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_POSTSOLVE_RELINKING_INPUT_VALIDATION_ATTR = (
    "_bayescatrack_postsolve_relinking_input_validation"
)


def install_postsolve_relinking_input_validation() -> None:
    """Install idempotent validation for relinking matrix/vector inputs."""

    from . import postsolve_relinking as postsolve_relinking_module

    original = postsolve_relinking_module.relink_tracks_at_geometry_issues
    if getattr(original, _POSTSOLVE_RELINKING_INPUT_VALIDATION_ATTR, False):
        return

    def _relink_tracks_at_geometry_issues_with_input_validation(
        track_rows: Any,
        issues: Sequence[Any],
        pairwise_costs: Mapping[tuple[int, int], np.ndarray],
        *,
        roi_indices_by_session: Sequence[Sequence[int]],
        config: Any = None,
    ) -> np.ndarray:
        _validate_roi_indices_by_session(roi_indices_by_session)
        if issues:
            _validate_pairwise_cost_matrices(pairwise_costs)
        return original(
            track_rows,
            issues,
            pairwise_costs,
            roi_indices_by_session=roi_indices_by_session,
            config=config,
        )

    setattr(
        _relink_tracks_at_geometry_issues_with_input_validation,
        _POSTSOLVE_RELINKING_INPUT_VALIDATION_ATTR,
        True,
    )
    setattr(
        _relink_tracks_at_geometry_issues_with_input_validation,
        "_bayescatrack_original",
        original,
    )
    postsolve_relinking_module.relink_tracks_at_geometry_issues = (
        _relink_tracks_at_geometry_issues_with_input_validation
    )


def _validate_roi_indices_by_session(
    roi_indices_by_session: Sequence[Sequence[int]],
) -> None:
    for session_index, values in enumerate(roi_indices_by_session):
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(
                f"roi_indices_by_session[{session_index}] must be one-dimensional"
            )


def _validate_pairwise_cost_matrices(
    pairwise_costs: Mapping[tuple[int, int], np.ndarray],
) -> None:
    for edge, matrix in pairwise_costs.items():
        array = np.asarray(matrix)
        if array.ndim != 2:
            raise ValueError(f"pairwise_costs[{edge!r}] must be two-dimensional")
