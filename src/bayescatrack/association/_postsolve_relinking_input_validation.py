"""Input-shape validation for post-solve relinking."""

from __future__ import annotations

import operator
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
        field_name = f"roi_indices_by_session[{session_index}]"
        array = np.asarray(values, dtype=object)
        if array.ndim != 1:
            raise ValueError(f"{field_name} must be one-dimensional")
        normalized = np.asarray(
            [_normalize_roi_index(value, field_name) for value in array.tolist()],
            dtype=int,
        )
        if len(set(normalized.tolist())) != normalized.size:
            raise ValueError(f"{field_name} must contain unique ROI indices")


def _normalize_roi_index(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain integer ROI indices")
    if isinstance(value, np.ndarray):
        raise ValueError(f"{field_name} must contain integer ROI indices")
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{field_name} must contain integer ROI indices")
        normalized = int(numeric)
    else:
        try:
            normalized = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{field_name} must contain integer ROI indices") from exc
    normalized = int(normalized)
    if normalized < 0:
        raise ValueError(f"{field_name} must contain non-negative ROI indices")
    return normalized


def _validate_pairwise_cost_matrices(
    pairwise_costs: Mapping[tuple[int, int], np.ndarray],
) -> None:
    for edge, matrix in pairwise_costs.items():
        array = np.asarray(matrix)
        if array.ndim != 2:
            raise ValueError(f"pairwise_costs[{edge!r}] must be two-dimensional")
