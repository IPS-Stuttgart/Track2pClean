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


def test_multisession_tracking_config_normalizes_integer_like_values():
    config = MultisessionTrackingConfig(
        max_session_gap="2",
        velocity_variance="25.0",
        regularization="0.0",
        start_cost="0.2",
        end_cost="0.3",
        gap_penalty="0.4",
        cost_threshold="6.0",
    )

    assert config.max_session_gap == 2
    assert config.velocity_variance == pytest.approx(25.0)
    assert config.regularization == pytest.approx(0.0)
    assert config.start_cost == pytest.approx(0.2)
    assert config.end_cost == pytest.approx(0.3)
    assert config.gap_penalty == pytest.approx(0.4)
    assert config.cost_threshold == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_session_gap", True),
        ("max_session_gap", 1.5),
        ("max_session_gap", 0),
        ("max_session_gap", np.nan),
        ("weighted_centroids", "false"),
        ("return_pairwise_components", 1),
        ("velocity_variance", True),
        ("velocity_variance", np.nan),
        ("velocity_variance", np.inf),
        ("velocity_variance", -1.0),
        ("regularization", False),
        ("regularization", np.nan),
        ("regularization", np.inf),
        ("regularization", -1.0),
        ("start_cost", True),
        ("start_cost", np.nan),
        ("start_cost", np.inf),
        ("start_cost", -1.0),
        ("end_cost", False),
        ("end_cost", np.nan),
        ("end_cost", np.inf),
        ("end_cost", -1.0),
        ("gap_penalty", True),
        ("gap_penalty", np.nan),
        ("gap_penalty", np.inf),
        ("gap_penalty", -1.0),
        ("cost_threshold", False),
        ("cost_threshold", np.nan),
        ("cost_threshold", np.inf),
        ("cost_threshold", -1.0),
        ("pairwise_cost_kwargs", (("large_cost", 1.0),)),
    ],
)
def test_multisession_tracking_config_rejects_invalid_controls(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        MultisessionTrackingConfig(**{field_name: value})


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
        del (
            pairwise_costs,
            session_sizes,
            start_cost,
            end_cost,
            gap_penalty,
            cost_threshold,
        )
        nonlocal calls
        calls += 1
        raise TypeError("internal solver bug")

    with pytest.raises(TypeError, match="internal solver bug"):
        _call_multisession_solver(solver, {}, [2, 3], MultisessionTrackingConfig())

    assert calls == 1
