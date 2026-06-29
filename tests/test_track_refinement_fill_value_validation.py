import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.track_refinement import (
    TrackGeometryIssue,
    TrackSmoothingConfig,
    smoothed_track_positions,
    split_tracks_at_issues,
)


class _IndexValueError:
    def __index__(self) -> int:
        raise ValueError("invalid index")


class _IndexOverflowError:
    def __index__(self) -> int:
        raise OverflowError("invalid index")


_BAD_FILL_VALUES = [
    True,
    np.bool_(False),
    np.asarray(-1),
    0,
    1,
    0.5,
    -1.5,
    np.nan,
    np.inf,
    "-1",
    _IndexValueError(),
    _IndexOverflowError(),
]


@pytest.mark.parametrize("bad_fill_value", _BAD_FILL_VALUES)
def test_track_smoothing_config_rejects_malformed_or_colliding_fill_value(
    bad_fill_value,
):
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        TrackSmoothingConfig(fill_value=bad_fill_value)


@pytest.mark.parametrize("bad_fill_value", _BAD_FILL_VALUES)
def test_smoothed_track_positions_rejects_malformed_or_colliding_fill_value(
    bad_fill_value,
):
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        smoothed_track_positions(
            np.array([[0, 1]], dtype=int),
            ({0: np.array([0.0, 0.0])}, {1: np.array([1.0, 1.0])}),
            fill_value=bad_fill_value,
        )


@pytest.mark.parametrize("bad_fill_value", _BAD_FILL_VALUES)
def test_split_tracks_at_issues_rejects_malformed_or_colliding_fill_value(
    bad_fill_value,
):
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        split_tracks_at_issues(
            np.array([[0, 1]], dtype=int),
            [],
            fill_value=bad_fill_value,
        )


def test_split_tracks_at_issues_rejects_array_valued_issue_index():
    issue = TrackGeometryIssue(
        track_index=np.asarray(0),
        session_index=0,
        roi_index=1,
        residual=9.0,
        robust_z=4.0,
        suggested_action="split_or_relink",
    )

    with pytest.raises(ValueError, match="issue.track_index must be an integer"):
        split_tracks_at_issues(
            np.array([[0, 1]], dtype=int),
            [issue],
        )


@pytest.mark.parametrize("bad_index", [_IndexValueError(), _IndexOverflowError()])
def test_split_tracks_at_issues_normalizes_issue_index_protocol_failures(bad_index):
    issue = TrackGeometryIssue(
        track_index=bad_index,
        session_index=0,
        roi_index=1,
        residual=9.0,
        robust_z=4.0,
        suggested_action="split_or_relink",
    )

    with pytest.raises(ValueError, match="issue.track_index must be an integer"):
        split_tracks_at_issues(
            np.array([[0, 1]], dtype=int),
            [issue],
        )


def test_split_tracks_at_issues_preserves_negative_missing_sentinel():
    rows = split_tracks_at_issues(
        np.array([[1, 2, 3]], dtype=int),
        [
            TrackGeometryIssue(
                track_index=0,
                session_index=1,
                roi_index=2,
                residual=9.0,
                robust_z=4.0,
                suggested_action="split_or_relink",
            )
        ],
        fill_value=np.int64(-9),
    )

    npt.assert_array_equal(rows, np.array([[1, -9, -9], [-9, 2, 3]], dtype=int))
