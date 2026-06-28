from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.track_refinement import (
    smoothed_track_positions,
    split_tracks_at_issues,
    track_geometry_issues,
)


def _position_tables():
    return (
        {0: np.array([0.0, 0.0])},
        {},
        {1: np.array([2.0, 2.0])},
    )


@pytest.mark.parametrize(
    "helper_name",
    ["track_geometry_issues", "smoothed_track_positions", "split_tracks_at_issues"],
)
def test_track_refinement_rejects_unconfigured_missing_token(helper_name):
    track_rows = np.array([[0, -2, 1]], dtype=int)

    with pytest.raises(ValueError, match="configured fill_value"):
        if helper_name == "track_geometry_issues":
            track_geometry_issues(track_rows, _position_tables())
        elif helper_name == "smoothed_track_positions":
            smoothed_track_positions(track_rows, _position_tables())
        else:
            split_tracks_at_issues(track_rows, ())


def test_smoothed_track_positions_allows_configured_missing_token():
    track_rows = np.array([[0, -2, 1]], dtype=int)

    smoothed = smoothed_track_positions(track_rows, _position_tables(), fill_value=-2)

    assert set(smoothed[0]) == {0, 2}
    npt.assert_allclose(smoothed[0][0], np.array([0.0, 0.0]), atol=1e-12)
    npt.assert_allclose(smoothed[0][2], np.array([2.0, 2.0]), atol=1e-12)
