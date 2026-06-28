from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

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


def test_track_matrix_edge_counter_keeps_stringified_integral_float_roi_labels():
    counter = track_matrix_edge_counter(
        np.asarray([["10.0", "20.0", "30.0"]], dtype=object)
    )

    assert counter[(0, 1, 10, 20)] == 1
    assert counter[(1, 2, 20, 30)] == 1
    assert sum(counter.values()) == 2


@pytest.mark.parametrize(
    "bad_value",
    [
        "not-a-roi",
        "1.5",
        "inf",
        Decimal("1.5"),
        Decimal("NaN"),
        Fraction(3, 2),
    ],
)
def test_track_matrix_edge_counter_rejects_malformed_roi_labels(bad_value):
    with pytest.raises(ValueError, match="non-integer ROI index"):
        track_matrix_edge_counter(np.asarray([[bad_value, 2]], dtype=object))


@pytest.mark.parametrize(
    ("roi_value", "expected_roi"),
    [
        (Decimal("10"), 10),
        (Fraction(20, 1), 20),
    ],
)
def test_track_matrix_edge_counter_accepts_exact_integral_roi_objects(
    roi_value,
    expected_roi,
):
    counter = track_matrix_edge_counter(np.asarray([[roi_value, 2]], dtype=object))

    assert counter[(0, 1, expected_roi, 2)] == 1


def test_consensus_track_rows_seeds_from_stringified_integral_float_roi_labels():
    track_rows = consensus_track_rows(
        ("s0", "s1"),
        [
            np.asarray([["10.0", "20.0"]], dtype=object),
            np.asarray([[10, 20]], dtype=object),
        ],
        min_votes=2,
    )

    np.testing.assert_array_equal(track_rows, np.asarray([[10, 20]], dtype=int))
