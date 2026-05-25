from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.experiments.solver_prior_tuning import (
    SolverPriorSearchConfig,
    parse_threshold_list,
)


def test_solver_prior_threshold_parser_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        parse_threshold_list("-0.1")


def test_solver_prior_threshold_parser_accepts_zero_and_none():
    assert parse_threshold_list("0,none") == (0.0, None)


def test_solver_prior_grid_rejects_negative_programmatic_thresholds():
    search = SolverPriorSearchConfig(
        start_costs=(1.0,),
        end_costs=(1.0,),
        gap_penalties=(0.0,),
        cost_thresholds=(-1.0,),
    )

    with pytest.raises(ValueError, match="non-negative"):
        search.candidates()


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("start_costs", True),
        ("end_costs", False),
        ("gap_penalties", np.bool_(True)),
        ("cost_thresholds", np.bool_(False)),
    ],
)
def test_solver_prior_grid_rejects_boolean_programmatic_values(field, bad_value):
    kwargs = {
        "start_costs": (1.0,),
        "end_costs": (1.0,),
        "gap_penalties": (0.0,),
        "cost_thresholds": (None,),
    }
    kwargs[field] = (bad_value,)
    search = SolverPriorSearchConfig(**kwargs)

    with pytest.raises(ValueError, match="boolean"):
        search.candidates()
