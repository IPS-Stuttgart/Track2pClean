from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from bayescatrack.association.track_refinement import smoothed_track_positions


def test_smoothed_track_positions_rejects_vector_track_rows():
    position_tables = (
        {0: np.asarray([0.0, 0.0])},
        {1: np.asarray([1.0, 1.0])},
    )

    with pytest.raises(ValueError, match="track_rows must be two-dimensional"):
        smoothed_track_positions(np.asarray([0, 1], dtype=int), position_tables)


@pytest.mark.parametrize(
    "position_tables",
    [
        ({0: np.asarray([0.0, 0.0])},),
        (
            {0: np.asarray([0.0, 0.0])},
            {1: np.asarray([1.0, 1.0])},
            {2: np.asarray([2.0, 2.0])},
        ),
    ],
)
def test_smoothed_track_positions_rejects_session_table_count_mismatch(
    position_tables,
):
    with pytest.raises(
        ValueError, match="position_tables must contain one table per session"
    ):
        smoothed_track_positions(np.asarray([[0, 1]], dtype=int), position_tables)


def test_smoothed_track_positions_keeps_valid_rows():
    rows = np.asarray([[0, 1]], dtype=int)
    position_tables = (
        {0: np.asarray([0.0, 0.0])},
        {1: np.asarray([2.0, 2.0])},
    )

    smoothed = smoothed_track_positions(rows, position_tables)

    assert set(smoothed) == {0}
    npt.assert_allclose(smoothed[0][0], np.asarray([0.0, 0.0]), atol=1e-12)
    npt.assert_allclose(smoothed[0][1], np.asarray([2.0, 2.0]), atol=1e-12)
