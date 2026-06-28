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


@pytest.mark.parametrize(
    "n_rois",
    (True, False, np.bool_(True), np.bool_(False), -1, 1.5, float("nan"), ""),
)
def test_absence_cost_vector_rejects_invalid_roi_count(n_rois: object) -> None:
    plane = _plane(0)
    plane.n_rois = n_rois

    with pytest.raises(ValueError, match=r"plane\.n_rois"):
        absence_cost_vector(plane)


def test_gap_penalty_matrix_rejects_invalid_roi_counts() -> None:
    reference = _plane(0)
    reference.n_rois = 1.5
    measurement = _plane(1)

    with pytest.raises(ValueError, match=r"reference_plane\.n_rois"):
        gap_penalty_matrix(reference, measurement)

    reference = _plane(1)
    measurement = _plane(0)
    measurement.n_rois = True

    with pytest.raises(ValueError, match=r"measurement_plane\.n_rois"):
        gap_penalty_matrix(reference, measurement)


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
def test_absence_model_config_rejects_boolean_scalars(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        AbsenceModelConfig(**{field: value})


@pytest.mark.parametrize(
    "value",
    (
        np.asarray([1.0, 2.0], dtype=float),
        np.asarray([], dtype=float),
        "",
        "1.0",
        object(),
    ),
)
def test_absence_model_config_rejects_non_scalar_controls(value: object) -> None:
    with pytest.raises(ValueError, match="base_absence_cost"):
        AbsenceModelConfig(base_absence_cost=value)


def test_absence_model_config_accepts_numpy_scalar_controls() -> None:
    config = AbsenceModelConfig(base_absence_cost=np.asarray(1.25))

    assert config.base_absence_cost == 1.25


def test_absence_cost_vector_ignores_nonfinite_cell_probability_entries() -> None:
    plane = _plane(
        4,
        cell_probabilities=np.asarray([1.0, np.nan, np.inf, 0.0], dtype=float),
    )

    costs = absence_cost_vector(
        plane,
        config=AbsenceModelConfig(
            base_absence_cost=1.0,
            low_cell_probability_discount=0.5,
            trace_missing_discount=0.0,
        ),
    )

    np.testing.assert_allclose(
        costs,
        np.asarray([1.0, 1.0, 1.0, 0.5], dtype=float),
    )
    assert np.all(np.isfinite(costs))


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


@pytest.mark.parametrize("session_gap", (1.5, np.float64(2.5), "1.5", ""))
def test_gap_penalty_matrix_rejects_fractional_session_gap(session_gap: object) -> None:
    reference = _plane(1)
    measurement = _plane(1)

    with pytest.raises(ValueError, match="session_gap"):
        gap_penalty_matrix(reference, measurement, session_gap=session_gap)


@pytest.mark.parametrize("session_gap", (2, 2.0, np.int64(2), "2"))
def test_gap_penalty_matrix_accepts_integer_like_session_gap(
    session_gap: object,
) -> None:
    reference = _plane(1)
    measurement = _plane(1)

    penalties = gap_penalty_matrix(reference, measurement, session_gap=session_gap)

    np.testing.assert_allclose(penalties, np.asarray([[1.0]], dtype=float))


def test_gap_penalty_matrix_rejects_invalid_absence_cost_vectors() -> None:
    reference = _plane(2)
    measurement = _plane(1)

    with pytest.raises(ValueError, match="reference_absence_costs"):
        gap_penalty_matrix(
            reference,
            measurement,
            reference_absence_costs=np.asarray([1.0, np.nan], dtype=float),
            measurement_absence_costs=np.asarray([1.0], dtype=float),
        )
    with pytest.raises(ValueError, match="measurement_absence_costs"):
        gap_penalty_matrix(
            reference,
            measurement,
            reference_absence_costs=np.asarray([1.0, 1.0], dtype=float),
            measurement_absence_costs=np.asarray([-0.1], dtype=float),
        )


def test_apply_absence_adjustment_rejects_broadcastable_shape_mismatch() -> None:
    reference = _plane(2)
    measurement = _plane(3)

    with pytest.raises(ValueError, match="cost_matrix shape must match"):
        apply_absence_adjustment(
            np.zeros((2, 1), dtype=float),
            reference,
            measurement,
        )


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
