from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.experiments import solver_prior_tuning as tuning


def _single_candidate_search(**overrides):
    settings = {
        "start_costs": (1.0,),
        "end_costs": (1.0,),
        "gap_penalties": (0.0,),
        "cost_thresholds": (None,),
    }
    settings.update(overrides)
    return tuning.SolverPriorSearchConfig(**settings)


def test_threshold_parser_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        tuning.parse_threshold_list("-0.1")


def test_threshold_parser_accepts_zero_and_none():
    assert tuning.parse_threshold_list("0,none") == (0.0, None)


def test_candidate_grid_rejects_negative_thresholds():
    search = _single_candidate_search(cost_thresholds=(-1.0,))

    with pytest.raises(ValueError, match="non-negative"):
        search.candidates()


def test_candidate_grid_rejects_boolean_start_costs():
    search = _single_candidate_search(start_costs=(True,))

    with pytest.raises(ValueError, match="boolean"):
        search.candidates()


def test_candidate_grid_rejects_numpy_boolean_thresholds():
    search = _single_candidate_search(cost_thresholds=(np.bool_(False),))

    with pytest.raises(ValueError, match="boolean"):
        search.candidates()
