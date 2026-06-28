from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bayescatrack.association.absence_model import (
    AbsenceModelConfig,
    absence_cost_vector,
    gap_penalty_matrix,
)


def _plane(n_rois: int) -> SimpleNamespace:
    return SimpleNamespace(
        n_rois=n_rois,
        cell_probabilities=None,
        traces=np.zeros((n_rois, 1), dtype=float),
        spike_traces=None,
    )


@pytest.mark.parametrize(
    "registered_empty_mask",
    (
        np.asarray(["off", "on"]),
        np.asarray(["0", "1"]),
        np.asarray([0, 2]),
        np.asarray([0.0, 0.5]),
        np.asarray([0.0, np.nan]),
    ),
)
def test_absence_cost_vector_rejects_non_binary_registered_empty_mask(
    registered_empty_mask: np.ndarray,
) -> None:
    plane = _plane(2)

    with pytest.raises(ValueError, match="registered_empty_mask"):
        absence_cost_vector(plane, registered_empty_mask=registered_empty_mask)


def test_absence_cost_vector_accepts_binary_numeric_registered_empty_mask() -> None:
    plane = _plane(2)

    costs = absence_cost_vector(
        plane,
        registered_empty_mask=np.asarray([1, 0], dtype=int),
        config=AbsenceModelConfig(
            base_absence_cost=1.0,
            empty_registered_mask_discount=0.25,
            trace_missing_discount=0.0,
        ),
    )

    np.testing.assert_allclose(costs, np.asarray([0.75, 1.0], dtype=float))


def test_gap_penalty_matrix_rejects_non_binary_registered_empty_mask() -> None:
    reference = _plane(1)
    measurement = _plane(2)

    with pytest.raises(ValueError, match="registered_empty_mask"):
        gap_penalty_matrix(
            reference,
            measurement,
            registered_empty_mask=np.asarray(["off", "on"]),
        )
