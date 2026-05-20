from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack import CalciumPlaneData, Track2pSession
from bayescatrack.multisession_tracking import (
    MultisessionTrackingConfig,
    _call_multisession_solver,
    track_sessions_multisession,
)


def test_single_session_multisession_tracking_short_circuits():
    plane = CalciumPlaneData(
        roi_masks=np.ones((1, 2, 2), dtype=bool), roi_indices=np.array([7])
    )
    session = Track2pSession(
        session_dir=Path("s1"),
        session_name="s1",
        session_date=None,
        plane_data=plane,
    )

    result = track_sessions_multisession([session])

    npt.assert_array_equal(result.track_matrix, np.array([[0]]))
    npt.assert_array_equal(result.track_roi_index_matrix, np.array([[7]]))
    assert result.summary()["n_tracks"] == 1


def test_call_multisession_solver_selects_signature_compatible_aliases():
    observed_kwargs = {}

    def solver(
        pairwise_costs,
        *,
        session_sizes,
        birth_cost,
        death_cost,
        gap_cost,
        cost_threshold,
    ):
        assert pairwise_costs == {}
        observed_kwargs.update(
            {
                "session_sizes": session_sizes,
                "birth_cost": birth_cost,
                "death_cost": death_cost,
                "gap_cost": gap_cost,
                "cost_threshold": cost_threshold,
            }
        )
        return "ok"

    config = MultisessionTrackingConfig()

    result = _call_multisession_solver(solver, {}, [2, 3], config)

    assert result == "ok"
    assert observed_kwargs == {
        "session_sizes": [2, 3],
        "birth_cost": config.start_cost,
        "death_cost": config.end_cost,
        "gap_cost": config.gap_penalty,
        "cost_threshold": config.cost_threshold,
    }


def test_call_multisession_solver_does_not_retry_internal_typeerror():
    calls = 0

    def solver(
        pairwise_costs,
        *,
        session_sizes,
        start_cost,
        end_cost,
        gap_penalty,
        cost_threshold,
    ):
        del pairwise_costs, session_sizes, start_cost, end_cost, gap_penalty, cost_threshold
        nonlocal calls
        calls += 1
        raise TypeError("internal solver bug")

    with pytest.raises(TypeError, match="internal solver bug"):
        _call_multisession_solver(solver, {}, [2, 3], MultisessionTrackingConfig())

    assert calls == 1
