import numpy as np
import pytest
from bayescatrack import load_suite2p_plane


def _write_suite2p_plane_with_invalid_overlap_pixel(plane_dir):
    plane_dir.mkdir()
    stat = np.empty(1, dtype=object)
    stat[0] = {
        "ypix": np.array([0, 5]),
        "xpix": np.array([0, 5]),
        "lam": np.array([1.0, 1.0]),
        "overlap": np.array([False, True]),
    }
    np.save(plane_dir / "stat.npy", stat)
    np.save(
        plane_dir / "ops.npy",
        {"Ly": 2, "Lx": 2, "meanImg": np.ones((2, 2), dtype=float)},
    )


def test_load_suite2p_plane_validation_rejects_out_of_bounds_overlap_pixels_before_exclusion(tmp_path):
    plane_dir = tmp_path / "plane0"
    _write_suite2p_plane_with_invalid_overlap_pixel(plane_dir)

    with pytest.raises(ValueError, match="within image bounds"):
        load_suite2p_plane(plane_dir)


def test_load_suite2p_plane_validation_rejects_out_of_bounds_overlap_pixels_when_retained(
    tmp_path,
):
    plane_dir = tmp_path / "plane0"
    _write_suite2p_plane_with_invalid_overlap_pixel(plane_dir)

    with pytest.raises(ValueError, match="within image bounds"):
        load_suite2p_plane(plane_dir, exclude_overlapping_pixels=False)
