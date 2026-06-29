from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bayescatrack.association.absence_model import (
    absence_cost_vector,
    gap_penalty_matrix,
)


def _plane(n_rois: int, *, cell_probabilities: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        n_rois=n_rois,
        cell_probabilities=cell_probabilities,
        traces=np.zeros((n_rois, 1), dtype=float),
        spike_traces=None,
    )


@pytest.mark.parametrize(
    ("plane_kwargs", "call_kwargs", "message"),
    (
        (
            {"cell_probabilities": np.asarray([[1.0], [0.0]], dtype=float)},
            {},
            "plane.cell_probabilities",
        ),
        ({}, {"registered_empty_mask": np.asarray([[True], [False]])}, "registered_empty_mask"),
        ({}, {"local_density": np.asarray([[0.0], [1.0]], dtype=float)}, "local_density"),
    ),
)
def test_absence_cost_vector_rejects_nested_optional_cues(
    plane_kwargs: dict[str, np.ndarray],
    call_kwargs: dict[str, np.ndarray],
    message: str,
) -> None:
    plane = _plane(2, **plane_kwargs)

    with pytest.raises(ValueError, match=message):
        absence_cost_vector(plane, **call_kwargs)


def test_gap_penalty_matrix_rejects_nested_explicit_absence_cost_vectors() -> None:
    reference = _plane(2)
    measurement = _plane(1)

    with pytest.raises(ValueError, match="absence cost vectors"):
        gap_penalty_matrix(
            reference,
            measurement,
            reference_absence_costs=np.asarray([[1.0], [1.0]], dtype=float),
            measurement_absence_costs=np.asarray([1.0], dtype=float),
        )
