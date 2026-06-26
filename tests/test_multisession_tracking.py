from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack import CalciumPlaneData, Track2pSession
from bayescatrack.multisession_tracking import (
    MultisessionTrackingConfig,
    _call_multisession_solver,
    _coerce_solver_tracks,
    _track_matrix_to_roi_index_matrix,
    track_sessions_multisession,
)


def _session(name: str, roi_indices: np.ndarray) -> Track2pSession:
    return Track2pSession(
        session_dir=Path(name),
        session_name=name,
        session_date=None,
        plane_data=CalciumPlaneData(
            roi_masks=np.ones((len(roi_indices), 2, 2), dtype=bool),
            roi_indices=roi_indices,
        ),
    )


def _two_session_fixture() -> list[Track2pSession]:
    return [
        _session("s1", np.array([10, 11])),
        _session("s2", np.array([20, 21])),
    ]


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


def test_multisession_tracking_maps_detection_indices_to_roi_indices():
    sessions = _two_session_fixture()

    def solver(pairwise_costs, **kwargs):
        assert (0, 1) in pairwise_costs
        assert kwargs["session_sizes"] == [2, 2]
        return {"tracks": [{0: 1, 1: 0}], "total_cost": 1.5}

    result = track_sessions_multisession(sessions, solver=solver)

    npt.assert_array_equal(result.track_matrix, np.array([[1, 0]]))
    npt.assert_array_equal(result.track_roi_index_matrix, np.array([[11, 20]]))
    assert result.total_cost == 1.5


def test_multisession_tracking_rejects_negative_solver_detection_index():
    sessions = _two_session_fixture()

    def solver(pairwise_costs, **kwargs):
        del pairwise_costs, kwargs
        return {"tracks": [{0: -1, 1: 0}]}

    with pytest.raises(ValueError, match="detection index must be a non-negative integer"):
        track_sessions_multisession(sessions, solver=solver)


def test_multisession_tracking_rejects_out_of_bounds_solver_detection_index():
    sessions = _two_session_fixture()

    def solver(pairwise_costs, **kwargs):
        del pairwise_costs, kwargs
        return {"tracks": [{0: 2, 1: 0}]}

    with pytest.raises(ValueError, match=r"outside 0\.\.1 for session 0"):
        track_sessions_multisession(sessions, solver=solver)


def test_multisession_tracking_rejects_boolean_solver_detection_index():
    sessions = _two_session_fixture()

    def solver(pairwise_costs, **kwargs):
        del pairwise_costs, kwargs
        return {"tracks": [{0: True, 1: 0}]}

    with pytest.raises(ValueError, match="detection index must be a non-negative integer"):
        track_sessions_multisession(sessions, solver=solver)


def test_track_matrix_to_roi_index_matrix_accepts_integer_like_float_indices():
    result = _track_matrix_to_roi_index_matrix(
        np.array([[1.0, -1.0]]),
        _two_session_fixture(),
    )

    npt.assert_array_equal(result, np.array([[11, -1]]))


@pytest.mark.parametrize(
    "bad_matrix",
    [
        np.array([[0.5, -1.0]]),
        np.array([[np.nan, -1.0]]),
        np.array([[True, False]]),
        np.array([["0", "-1"]], dtype=object),
    ],
)
def test_track_matrix_to_roi_index_matrix_rejects_ambiguous_detection_indices(
    bad_matrix,
):
    with pytest.raises(
        ValueError,
        match="track_matrix must contain integer detection indices",
    ):
        _track_matrix_to_roi_index_matrix(bad_matrix, _two_session_fixture())


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


def test_coerce_solver_tracks_accepts_integer_like_numpy_indices():
    tracks, total_cost = _coerce_solver_tracks(
        {"tracks": [{np.int64(0): np.float64(2.0)}], "total_cost": 1.25}
    )

    assert tracks == ({0: 2},)
    assert total_cost == 1.25


@pytest.mark.parametrize(
    ("bad_track", "message"),
    [
        ({True: 0}, "session index"),
        ({0: True}, "detection index"),
        ({0: -1}, "detection index"),
        ({0: 1.5}, "detection index"),
        ({"0": 0}, "session index"),
        ({0: "1"}, "detection index"),
        ({0: np.nan}, "detection index"),
    ],
)
def test_coerce_solver_tracks_rejects_malformed_indices(bad_track, message):
    with pytest.raises(ValueError, match=message):
        _coerce_solver_tracks([bad_track])
