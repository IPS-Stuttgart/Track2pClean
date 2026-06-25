from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.absence_model import (
    AbsenceModelConfig,
    absence_cost_vector,
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


@pytest.mark.parametrize(
    "field",
    (
        "base_absence_cost",
        "out_of_fov_discount",
        "low_cell_probability_discount",
        "empty_registered_mask_discount",
        "high_local_density_discount",
        "trace_missing_discount",
        "min_cost",
    ),
)
@pytest.mark.parametrize("value", (True, False, np.bool_(True), np.bool_(False)))
def test_absence_model_config_rejects_boolean_scalars(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        AbsenceModelConfig(**{field: value})


def test_absence_cost_vector_ignores_nonfinite_local_density_entries() -> None:
    plane = _plane(4)

    costs = absence_cost_vector(
        plane,
        local_density=np.asarray([0.0, np.nan, np.inf, -np.inf], dtype=float),
        config=AbsenceModelConfig(
            base_absence_cost=1.0,
            high_local_density_discount=0.25,
            trace_missing_discount=0.0,
        ),
    )

    np.testing.assert_allclose(costs, np.asarray([1.0, 1.0, 0.75, 1.0], dtype=float))


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
