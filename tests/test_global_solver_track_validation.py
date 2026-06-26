from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from bayescatrack.association import pyrecest_global_assignment as assignment


def _sessions() -> tuple[SimpleNamespace, SimpleNamespace]:
    return (
        SimpleNamespace(
            plane_data=SimpleNamespace(n_rois=2, roi_indices=np.asarray([10, 11]))
        ),
        SimpleNamespace(
            plane_data=SimpleNamespace(n_rois=2, roi_indices=np.asarray([20, 21]))
        ),
    )


def test_tracks_to_suite2p_index_matrix_normalizes_solver_track_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_tracks_to_index_matrix(
        tracks: list[dict[int, int]],
        *,
        session_sizes: tuple[int, ...],
        fill_value: int,
    ) -> np.ndarray:
        observed["tracks"] = tuple(dict(track) for track in tracks)
        observed["session_sizes"] = tuple(session_sizes)
        observed["fill_value"] = fill_value
        return np.asarray([[1, 0]], dtype=int)

    monkeypatch.setattr(
        assignment,
        "_load_pyrecest_tracks_to_index_matrix",
        lambda: fake_tracks_to_index_matrix,
    )

    matrix = assignment.tracks_to_suite2p_index_matrix(
        ({np.int64(0): np.float64(1.0), 1: 0},),
        _sessions(),
    )

    assert observed == {
        "tracks": ({0: 1, 1: 0},),
        "session_sizes": (2, 2),
        "fill_value": -1,
    }
    np.testing.assert_array_equal(matrix, np.asarray([[11, 20]], dtype=object))


@pytest.mark.parametrize(
    ("tracks", "message"),
    [
        (({0: True, 1: 0},), "detection index"),
        (({True: 0, 0: 0},), "session index"),
        (({0: 1.25, 1: 0},), "detection index"),
        (({0: np.nan, 1: 0},), "detection index"),
        (({0: "1", 1: 0},), "detection index"),
        ((("0", 1),), "mapping"),
        (({"0": 1, 1: 0},), "session index"),
        (({0: -1, 1: 0},), "detection index"),
        (({2: 0},), "session index 2 out of bounds"),
        (({0: 2},), "detection index 2 out of bounds"),
    ],
)
def test_tracks_to_suite2p_index_matrix_rejects_malformed_solver_tracks(
    monkeypatch: pytest.MonkeyPatch,
    tracks: object,
    message: str,
) -> None:
    def forbidden_tracks_to_index_matrix(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("converter should not be called for malformed tracks")

    monkeypatch.setattr(
        assignment,
        "_load_pyrecest_tracks_to_index_matrix",
        lambda: forbidden_tracks_to_index_matrix,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        assignment.tracks_to_suite2p_index_matrix(tracks, _sessions())


@pytest.mark.parametrize(
    "size",
    [
        True,
        np.bool_(False),
        1.5,
        np.nan,
        np.inf,
        "2",
        b"2",
        -1,
        np.asarray(2),
        object(),
    ],
)
def test_tracks_to_suite2p_index_matrix_rejects_malformed_session_sizes(
    monkeypatch: pytest.MonkeyPatch,
    size: object,
) -> None:
    def forbidden_tracks_to_index_matrix(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("converter should not be called for malformed session sizes")

    monkeypatch.setattr(
        assignment,
        "_load_pyrecest_tracks_to_index_matrix",
        lambda: forbidden_tracks_to_index_matrix,
    )
    malformed_sessions = (
        SimpleNamespace(
            plane_data=SimpleNamespace(
                n_rois=size,
                roi_indices=np.asarray([], dtype=int),
            )
        ),
    )

    with pytest.raises(ValueError, match="plane_data.n_rois"):
        assignment.tracks_to_suite2p_index_matrix(({0: 0},), malformed_sessions)
