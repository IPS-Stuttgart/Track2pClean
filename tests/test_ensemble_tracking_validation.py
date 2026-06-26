from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.ensemble_tracking import (
    consensus_edge_counter,
    consensus_track_rows,
    track_matrix_edge_counter,
)


@pytest.mark.parametrize(
    "min_votes", [0, -1, 1.5, np.inf, np.nan, True, np.bool_(True)]
)
def test_consensus_edge_counter_rejects_invalid_min_votes(min_votes):
    with pytest.raises(ValueError, match="min_votes"):
        consensus_edge_counter([[[0, 1]]], min_votes=min_votes)


@pytest.mark.parametrize(
    "session_pairs",
    [
        [(False, 1)],
        [(0, np.bool_(True))],
        [(0, 1.5)],
        [(0, np.nan)],
    ],
)
def test_track_matrix_edge_counter_rejects_malformed_session_pairs(session_pairs):
    with pytest.raises(ValueError, match="session_pairs"):
        track_matrix_edge_counter([[0, 1, 2]], session_pairs=session_pairs)


def test_consensus_track_rows_rejects_fractional_start_session_index():
    with pytest.raises(ValueError, match="start_session_index"):
        consensus_track_rows(
            ("s0", "s1"),
            [[[0, 1]]],
            start_session_index=0.5,
        )


def test_consensus_edge_counter_accepts_integer_like_control_strings():
    counter = consensus_edge_counter(
        [[[0, 1, 2]]],
        min_votes="1",
        session_pairs=[("0", "2")],
    )

    assert counter[(0, 2, 0, 2)] == 1
