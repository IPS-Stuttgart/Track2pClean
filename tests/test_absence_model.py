from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.absence_model import (
    AbsenceModelConfig,
    absence_cost_vector,
    absence_summary,
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


def test_absence_config_rejects_boolean_numeric_values() -> None:
    with pytest.raises(
        ValueError,
        match="base_absence_cost must be finite and non-negative",
    ):
        AbsenceModelConfig(base_absence_cost=True)


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


def test_absence_cost_vector_sanitizes_nonfinite_cell_probabilities() -> None:
    plane = _plane(4, cell_probabilities=np.asarray([np.nan, np.inf, -np.inf, 0.25]))
    config = AbsenceModelConfig(
        base_absence_cost=1.0,
        low_cell_probability_discount=0.5,
        trace_missing_discount=0.0,
    )

    costs = absence_cost_vector(plane, config=config)

    assert np.all(np.isfinite(costs))
    assert costs[0] == pytest.approx(1.0)
    assert costs[1] == pytest.approx(1.0)
    assert costs[2] == pytest.approx(0.5)
    assert costs[3] == pytest.approx(0.625)


def test_absence_cost_vector_sanitizes_nonfinite_local_density() -> None:
    plane = _plane(4)
    config = AbsenceModelConfig(
        base_absence_cost=2.0,
        high_local_density_discount=1.0,
        trace_missing_discount=0.0,
    )

    costs = absence_cost_vector(
        plane,
        local_density=np.asarray([np.nan, np.inf, -np.inf, 10.0]),
        config=config,
    )

    assert np.all(np.isfinite(costs))
    assert costs[:3] == pytest.approx([2.0, 2.0, 2.0])
    assert costs[3] == pytest.approx(1.0)


def test_gap_penalty_matrix_rejects_nonfinite_explicit_absence_costs() -> None:
    reference = _plane(2)
    measurement = _plane(2)

    with pytest.raises(
        ValueError,
        match="reference_absence_costs must contain only finite values",
    ):
        gap_penalty_matrix(
            reference,
            measurement,
            session_gap=2,
            reference_absence_costs=np.asarray([0.0, np.nan]),
            measurement_absence_costs=np.asarray([0.0, 0.0]),
        )


def test_apply_absence_adjustment_rejects_nonfinite_base_cost_matrix() -> None:
    reference = _plane(1)
    measurement = _plane(1)

    with pytest.raises(
        ValueError,
        match="absence-adjusted cost matrix must contain only finite values",
    ):
        apply_absence_adjustment(
            np.asarray([[np.nan]]),
            reference,
            measurement,
            session_gap=1,
        )


def test_absence_summary_rejects_nonfinite_explicit_costs() -> None:
    with pytest.raises(ValueError, match="costs must contain only finite values"):
        absence_summary(_plane(2), costs=np.asarray([0.0, np.inf]))
