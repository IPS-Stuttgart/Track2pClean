from __future__ import annotations

import inspect

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.activity_tie_breaker import (
    activity_tie_breaker_cost_matrix,
)
from bayescatrack.association.pyrecest_global_assignment import (
    build_registered_pairwise_costs,
    solve_global_assignment_for_sessions,
)


def test_activity_tie_breaker_cost_matrix_scales_selected_component() -> None:
    pairwise_components = {
        "activity_tiebreaker_cost": np.array([[0.0, 0.5], [1.0, np.nan]], dtype=float),
        "activity_tiebreaker_available": np.array([[1.0, 1.0], [1.0, 0.0]]),
    }

    tie_breaker_cost = activity_tie_breaker_cost_matrix(pairwise_components, weight=0.1)

    npt.assert_allclose(tie_breaker_cost, np.array([[-0.05, 0.0], [0.05, 0.0]]))


def test_activity_tie_breaker_cost_matrix_ignores_missing_activity() -> None:
    pairwise_components = {
        "activity_tiebreaker_cost": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        "activity_tiebreaker_available": np.zeros((2, 2), dtype=float),
    }

    tie_breaker_cost = activity_tie_breaker_cost_matrix(pairwise_components, weight=0.1)

    npt.assert_allclose(tie_breaker_cost, np.zeros((2, 2), dtype=float))


def test_activity_tie_breaker_cost_matrix_can_gate_noncompetitive_edges() -> None:
    pairwise_components = {
        "activity_tiebreaker_cost": np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=float
        ),
        "activity_tiebreaker_available": np.ones((2, 3), dtype=float),
    }
    base_cost_matrix = np.array([[1.0, 1.2, 5.0], [2.0, 2.6, 2.1]], dtype=float)

    tie_breaker_cost = activity_tie_breaker_cost_matrix(
        pairwise_components,
        weight=0.1,
        base_cost_matrix=base_cost_matrix,
        max_row_margin=0.25,
    )

    npt.assert_allclose(
        tie_breaker_cost,
        np.array([[-0.05, 0.05, 0.0], [0.05, 0.0, 0.05]], dtype=float),
    )


def test_activity_tie_breaker_cost_matrix_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        activity_tie_breaker_cost_matrix(
            {"activity_tiebreaker_cost": np.zeros((1, 1))}, weight=-1.0
        )
    with pytest.raises(KeyError, match="missing"):
        activity_tie_breaker_cost_matrix({}, component_name="missing")
    with pytest.raises(ValueError, match="two-dimensional"):
        activity_tie_breaker_cost_matrix(
            {"activity_tiebreaker_cost": np.zeros((1, 1, 1))}
        )


def test_global_assignment_exposes_activity_tie_breaker_parameters() -> None:
    for function in (
        build_registered_pairwise_costs,
        solve_global_assignment_for_sessions,
    ):
        signature = inspect.signature(function)
        assert "activity_tie_breaker_weight" in signature.parameters
        assert "activity_tie_breaker_component" in signature.parameters
        assert "activity_tie_breaker_neutral_cost" in signature.parameters
        assert "activity_tie_breaker_availability_component" in signature.parameters
        assert "activity_tie_breaker_max_row_margin" in signature.parameters
        assert "activity_tie_breaker_max_column_margin" in signature.parameters
        assert "activity_trace_source" in signature.parameters
        assert "activity_event_threshold" in signature.parameters
        assert signature.parameters["activity_tie_breaker_weight"].default == 0.0
        assert (
            signature.parameters["activity_tie_breaker_component"].default
            == "activity_tiebreaker_cost"
        )
        assert (
            signature.parameters["activity_tie_breaker_neutral_cost"].default == 0.5
        )
        assert (
            signature.parameters[
                "activity_tie_breaker_availability_component"
            ].default == "activity_tiebreaker_available"
        )
