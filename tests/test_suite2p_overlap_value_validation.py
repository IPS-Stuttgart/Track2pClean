from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import load_suite2p_plane, load_track2p_subject


def _write_suite2p_plane_with_overlap(plane_dir, overlap: np.ndarray) -> None:
    plane_dir.mkdir(parents=True)
    stat = np.empty(1, dtype=object)
    stat[0] = {
        "ypix": np.asarray([0, 1], dtype=int),
        "xpix": np.asarray([0, 1], dtype=int),
        "lam": np.asarray([1.0, 1.0], dtype=float),
        "overlap": overlap,
    }
    np.save(plane_dir / "stat.npy", stat)
    np.save(
        plane_dir / "ops.npy",
        {"Ly": 2, "Lx": 2, "meanImg": np.ones((2, 2), dtype=float)},
    )


def test_load_suite2p_plane_rejects_string_overlap_values(tmp_path):
    plane_dir = tmp_path / "plane0"
    _write_suite2p_plane_with_overlap(
        plane_dir,
        np.asarray(["False", "True"], dtype=object),
    )

    with pytest.raises(ValueError, match="overlap must contain boolean values"):
        load_suite2p_plane(plane_dir)


def test_load_track2p_subject_rejects_string_overlap_values(tmp_path):
    subject_dir = tmp_path / "jm123"
    plane_dir = subject_dir / "2024-05-01_a" / "suite2p" / "plane0"
    _write_suite2p_plane_with_overlap(
        plane_dir,
        np.asarray(["False", "True"], dtype=object),
    )

    with pytest.raises(ValueError, match="overlap must contain boolean values"):
        load_track2p_subject(subject_dir, input_format="suite2p")


def test_load_suite2p_plane_accepts_object_boolean_overlap_values(tmp_path):
    plane_dir = tmp_path / "plane0"
    _write_suite2p_plane_with_overlap(
        plane_dir,
        np.asarray([False, np.bool_(True)], dtype=object),
    )

    plane = load_suite2p_plane(plane_dir)

    assert plane.n_rois == 1
    assert int(np.count_nonzero(plane.roi_masks[0])) == 1
    assert bool(plane.roi_masks[0, 0, 0])
