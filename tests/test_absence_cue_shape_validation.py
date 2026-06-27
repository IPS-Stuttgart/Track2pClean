from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.absence_model import (
    absence_cost_vector,
    gap_penalty_matrix,
)


def _plane(
    n_rois: int,
    *,
    cell_probabilities: np.ndarray | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        n_rois=n_rois,
        cell_probabilities=cell_probabilities,
        traces=np.zeros((n_rois, 1), dtype=float),
        spike_traces=None,
    )


def test_absence_cost_vector_rejects_cell_probability_length_mismatch() -> None:
    plane = _plane(2, cell_probabilities=np.asarray([1.0], dtype=float))

    with pytest.raises(ValueError, match="plane.cell_probabilities"):
        absence_cost_vector(plane)


@pytest.mark.parametrize(
    ("kwarg", "value"),
    (
        ("registered_empty_mask", np.asarray([True], dtype=bool)),
        ("local_density", np.asarray([0.5], dtype=float)),
    ),
)
def test_absence_cost_vector_rejects_optional_cue_length_mismatch(
    kwarg: str,
    value: np.ndarray,
) -> None:
    plane = _plane(2)

    with pytest.raises(ValueError, match=kwarg):
        absence_cost_vector(plane, **{kwarg: value})


def test_gap_penalty_matrix_rejects_registered_empty_mask_length_mismatch() -> None:
    reference = _plane(1)
    measurement = _plane(2)

    with pytest.raises(ValueError, match="registered_empty_mask"):
        gap_penalty_matrix(
            reference,
            measurement,
            registered_empty_mask=np.asarray([True], dtype=bool),
        )


def test_gap_penalty_matrix_rejects_measurement_density_length_mismatch() -> None:
    reference = _plane(1)
    measurement = _plane(2)

    with pytest.raises(ValueError, match="local_density"):
        gap_penalty_matrix(
            reference,
            measurement,
            measurement_local_density=np.asarray([0.5], dtype=float),
        )
