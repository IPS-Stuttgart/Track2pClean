from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.testing as npt
import pytest

from bayescatrack.association.track_refinement import (
    roi_position_tables_from_sessions,
    split_tracks_at_issues,
    smoothed_track_positions,
)


@dataclass(frozen=True)
class _Plane:
    centroid_matrix: np.ndarray
    roi_indices: object | None = None
    n_rois: int | None = None

    def __post_init__(self) -> None:
        if self.n_rois is None:
            object.__setattr__(self, "n_rois", int(np.asarray(self.centroid_matrix).shape[1]))

    def centroids(self, *, order="xy", weighted=False):  # pylint: disable=unused-argument
        return self.centroid_matrix


@dataclass(frozen=True)
class _Session:
    plane_data: _Plane


def _session(centroids, *, roi_indices=None, n_rois=None):
    return _Session(
        _Plane(np.asarray(centroids, dtype=float), roi_indices=roi_indices, n_rois=n_rois)
    )


def test_roi_position_tables_preserve_explicit_roi_indices():
    tables = roi_position_tables_from_sessions(
        [_session([[1.0, 2.0], [3.0, 4.0]], roi_indices=[7, 11])]
    )

    assert set(tables[0]) == {7, 11}
    npt.assert_allclose(tables[0][7], np.asarray([1.0, 3.0]))
    npt.assert_allclose(tables[0][11], np.asarray([2.0, 4.0]))


def test_roi_position_tables_reject_short_roi_index_vector():
    with pytest.raises(ValueError, match="one entry per centroid"):
        roi_position_tables_from_sessions(
            [_session([[1.0, 2.0], [3.0, 4.0]], roi_indices=[7])]
        )


@pytest.mark.parametrize(
    ("roi_indices", "message"),
    [
        ([7, 7], "unique ROI indices"),
        ([True, 8], "must be an integer"),
        ([7, 8.5], "must be an integer"),
        ([7, -1], "non-negative ROI indices"),
    ],
)
def test_roi_position_tables_reject_malformed_roi_indices(roi_indices, message):
    with pytest.raises(ValueError, match=message):
        roi_position_tables_from_sessions(
            [_session([[1.0, 2.0], [3.0, 4.0]], roi_indices=roi_indices)]
        )


def test_roi_position_tables_rejects_n_roi_centroid_mismatch():
    with pytest.raises(ValueError, match="n_rois must match"):
        roi_position_tables_from_sessions(
            [_session([[1.0, 2.0], [3.0, 4.0]], n_rois=3)]
        )


def test_smoothed_track_positions_rejects_vector_track_rows():
    position_tables = (
        {0: np.asarray([0.0, 0.0])},
        {1: np.asarray([1.0, 1.0])},
    )

    with pytest.raises(ValueError, match="track_rows must be two-dimensional"):
        smoothed_track_positions(np.asarray([0, 1], dtype=int), position_tables)


@pytest.mark.parametrize(
    "track_rows",
    [
        np.asarray([[0.0, 1.5]], dtype=float),
        np.asarray([[0, np.nan]], dtype=float),
        np.asarray([[True, False]], dtype=bool),
        np.asarray([["0", "1"]], dtype=str),
    ],
)
def test_smoothed_track_positions_rejects_non_integer_track_rows(track_rows):
    position_tables = (
        {0: np.asarray([0.0, 0.0])},
        {1: np.asarray([1.0, 1.0])},
    )

    with pytest.raises(ValueError, match="finite integer ROI indices"):
        smoothed_track_positions(track_rows, position_tables)


def test_split_tracks_at_issues_rejects_non_integer_track_rows():
    with pytest.raises(ValueError, match="finite integer ROI indices"):
        split_tracks_at_issues(np.asarray([[0.0, 1.5]], dtype=float), ())


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
