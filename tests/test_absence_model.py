from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.absence_model import (
    AbsenceModelConfig,
    apply_absence_adjustment,
    gap_penalty_matrix,
)


def _plane(
    n_rois: int, *, cell_probabilities: np.ndarray | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        n_rois=n_rois,
        cell_probabilities=cell_probabilities,
        traces=np.zeros((n_rois, 1), dtype=float),
        spike_traces=None,
    )


def test_absence_model_config_rejects_nonfinite_discounts() -> None:
    with pytest.raises(ValueError, match="low_cell_probability_discount"):
        AbsenceModelConfig(low_cell_probability_discount=float("nan"))
    with pytest.raises(ValueError, match="empty_registered_mask_discount"):
        AbsenceModelConfig(empty_registered_mask_discount=float("inf"))


def test_absence_model_config_rejects_negative_discounts() -> None:
    with pytest.raises(ValueError, match="trace_missing_discount"):
        AbsenceModelConfig(trace_missing_discount=-0.1)


def test_gap_penalty_matrix_rejects_invalid_session_gap() -> None:
    reference = _plane(1)
    measurement = _plane(1)

    with pytest.raises(ValueError, match="session_gap"):
        gap_penalty_matrix(reference, measurement, session_gap=float("nan"))
    with pytest.raises(ValueError, match="session_gap"):
        gap_penalty_matrix(reference, measurement, session_gap=0)
    with pytest.raises(ValueError, match="session_gap"):
        gap_penalty_matrix(reference, measurement, session_gap=True)


def test_apply_absence_adjustment_uses_validated_gap_offset() -> None:
    reference = _plane(2, cell_probabilities=np.asarray([1.0, 0.0]))
    measurement = _plane(1, cell_probabilities=np.asarray([1.0]))
    base_costs = np.asarray([[1.0], [2.0]], dtype=float)

    adjusted = apply_absence_adjustment(
        base_costs,
        reference,
        measurement,
        session_gap=3,
        config=AbsenceModelConfig(
            base_absence_cost=1.0,
            low_cell_probability_discount=0.5,
            trace_missing_discount=0.0,
        ),
    )

    np.testing.assert_allclose(adjusted, np.asarray([[3.0], [3.5]], dtype=float))
