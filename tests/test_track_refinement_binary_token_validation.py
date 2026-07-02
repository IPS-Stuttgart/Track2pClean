import numpy as np
import pytest
from bayescatrack.association.track_refinement import (
    smoothed_track_positions,
    split_tracks_at_issues,
    track_geometry_issues,
)

_ERROR_MESSAGE = "track_rows must contain finite integer ROI indices"
_POSITION_TABLES = (
    {0: np.array([0.0, 0.0])},
    {1: np.array([1.0, 1.0])},
)


def _object_track_rows_with(binary_token):
    rows = np.empty((1, 2), dtype=object)
    rows[0, 0] = 0
    rows[0, 1] = binary_token
    return rows


@pytest.mark.parametrize("binary_token", [bytearray(b"1"), memoryview(b"1")])
def test_smoothed_track_positions_rejects_binary_text_like_track_row_tokens(
    binary_token,
):
    with pytest.raises(ValueError, match=_ERROR_MESSAGE):
        smoothed_track_positions(
            _object_track_rows_with(binary_token),
            _POSITION_TABLES,
        )


@pytest.mark.parametrize("binary_token", [bytearray(b"1"), memoryview(b"1")])
def test_track_geometry_issues_rejects_binary_text_like_track_row_tokens(binary_token):
    with pytest.raises(ValueError, match=_ERROR_MESSAGE):
        track_geometry_issues(
            _object_track_rows_with(binary_token),
            _POSITION_TABLES,
        )


@pytest.mark.parametrize("binary_token", [bytearray(b"1"), memoryview(b"1")])
def test_split_tracks_at_issues_rejects_binary_text_like_track_row_tokens(binary_token):
    with pytest.raises(ValueError, match=_ERROR_MESSAGE):
        split_tracks_at_issues(
            _object_track_rows_with(binary_token),
            [],
        )
