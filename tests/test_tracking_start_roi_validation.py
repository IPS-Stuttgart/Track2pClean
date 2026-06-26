from __future__ import annotations

from pathlib import Path

import bayescatrack.tracking as tracking
import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.tracking import run_registered_subject_tracking


def _write_single_session_subject(subject_dir: Path) -> None:
    plane_dir = subject_dir / "2024-05-01_a" / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True)
    roi_masks = np.zeros((2, 4, 4), dtype=bool)
    roi_masks[0, 0:2, 0:2] = True
    roi_masks[1, 2:4, 2:4] = True
    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(plane_dir / "F.npy", np.ones((2, 2), dtype=float))
    np.save(plane_dir / "fov.npy", np.zeros((4, 4), dtype=float))


def test_tracking_start_roi_restriction_preserves_valid_seed_order():
    track_rows = np.asarray([[0, 10], [1, 11]], dtype=int)

    restricted = tracking._restrict_track_rows_to_start_rois(
        track_rows,
        start_roi_indices=np.asarray([1, 3], dtype=np.int64),
        start_session_index=0,
        fill_value=-1,
    )

    npt.assert_array_equal(restricted, np.asarray([[1, 11], [3, -1]], dtype=int))


@pytest.mark.parametrize(
    "bad_start_roi_indices",
    [
        [True],
        [1.5],
        [np.inf],
        [-1],
        [0, 0],
        "0",
    ],
)
def test_tracking_start_roi_restriction_rejects_invalid_seed_indices(
    bad_start_roi_indices,
):
    track_rows = np.asarray([[0, 10], [1, 11]], dtype=int)

    with pytest.raises(ValueError, match="start_roi_indices"):
        tracking._restrict_track_rows_to_start_rois(
            track_rows,
            start_roi_indices=bad_start_roi_indices,
            start_session_index=0,
            fill_value=-1,
        )


@pytest.mark.parametrize("bad_start_session_index", [True, 0.5, np.nan, "0"])
def test_tracking_start_roi_restriction_rejects_invalid_start_session_index(
    bad_start_session_index,
):
    track_rows = np.asarray([[0, 10], [1, 11]], dtype=int)

    with pytest.raises(ValueError, match="start_session_index"):
        tracking._restrict_track_rows_to_start_rois(
            track_rows,
            start_roi_indices=[0, 1],
            start_session_index=bad_start_session_index,
            fill_value=-1,
        )


@pytest.mark.parametrize("bad_start_session_index", [-1, 2])
def test_tracking_start_roi_restriction_rejects_out_of_bounds_start_session_index(
    bad_start_session_index,
):
    track_rows = np.asarray([[0, 10], [1, 11]], dtype=int)

    with pytest.raises(IndexError, match="start_session_index"):
        tracking._restrict_track_rows_to_start_rois(
            track_rows,
            start_roi_indices=[0, 1],
            start_session_index=bad_start_session_index,
            fill_value=-1,
        )


def test_run_registered_subject_tracking_rejects_invalid_single_session_seed_indices(
    tmp_path: Path,
):
    subject_dir = tmp_path / "jm271"
    _write_single_session_subject(subject_dir)

    with pytest.raises(ValueError, match="start_roi_indices"):
        run_registered_subject_tracking(
            subject_dir,
            plane_name="plane0",
            input_format="auto",
            include_behavior=False,
            start_roi_indices=[0, 0],
        )


def test_run_registered_subject_tracking_rejects_fractional_start_session_index(
    tmp_path: Path,
):
    subject_dir = tmp_path / "jm271"
    _write_single_session_subject(subject_dir)

    with pytest.raises(ValueError, match="start_session_index"):
        run_registered_subject_tracking(
            subject_dir,
            plane_name="plane0",
            input_format="auto",
            include_behavior=False,
            start_session_index=0.5,
        )
