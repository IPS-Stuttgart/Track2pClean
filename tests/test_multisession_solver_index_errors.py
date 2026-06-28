from __future__ import annotations

import pytest
from bayescatrack.multisession_tracking import _coerce_solver_tracks, _tracks_to_matrix


class ValueErrorIndex:
    def __index__(self) -> int:
        raise ValueError("bad index")


class OverflowIndex:
    def __index__(self) -> int:
        raise OverflowError("bad index")


@pytest.mark.parametrize("bad_index", [ValueErrorIndex(), OverflowIndex()])
def test_coerce_solver_tracks_normalizes_custom_session_index_errors(bad_index: object) -> None:
    with pytest.raises(
        ValueError,
        match="multisession solver track 0 session index must be a non-negative integer",
    ):
        _coerce_solver_tracks([{bad_index: 0}])


@pytest.mark.parametrize("bad_index", [ValueErrorIndex(), OverflowIndex()])
def test_coerce_solver_tracks_normalizes_custom_detection_index_errors(bad_index: object) -> None:
    with pytest.raises(
        ValueError,
        match="multisession solver track 0 detection index must be a non-negative integer",
    ):
        _coerce_solver_tracks([{0: bad_index}])


@pytest.mark.parametrize("bad_index", [ValueErrorIndex(), OverflowIndex()])
def test_tracks_to_matrix_normalizes_custom_session_count_index_errors(bad_index: object) -> None:
    with pytest.raises(ValueError, match="n_sessions must be a non-negative integer"):
        _tracks_to_matrix(({0: 0},), bad_index)
