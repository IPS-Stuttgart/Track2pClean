from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import load_suite2p_plane, load_track2p_subject


def test_load_track2p_subject_rejects_noninteger_suite2p_coordinates(tmp_path):
    subject_dir = tmp_path / "subject"
    plane_dir = subject_dir / "2024-01-01_session" / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True)

    stat = np.asarray(
        [
            {
                "ypix": np.asarray([1.5, 2.0], dtype=float),
                "xpix": np.asarray([1.0, 2.0], dtype=float),
                "lam": np.asarray([1.0, 1.0], dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "ops.npy", {"Ly": 4, "Lx": 5})

    with pytest.raises(
        ValueError, match="finite non-negative integer pixel coordinates"
    ):
        load_track2p_subject(
            subject_dir,
            input_format="suite2p",
            plane_name="plane0",
            load_traces=False,
            load_spike_traces=False,
        )


def test_load_track2p_subject_rejects_negative_suite2p_coordinates(tmp_path):
    subject_dir = tmp_path / "subject"
    plane_dir = subject_dir / "2024-01-01_session" / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True)

    stat = np.asarray(
        [
            {
                "ypix": np.asarray([1, 2], dtype=int),
                "xpix": np.asarray([1, -1], dtype=int),
                "lam": np.asarray([1.0, 1.0], dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "ops.npy", {"Ly": 4, "Lx": 5})

    with pytest.raises(
        ValueError, match="finite non-negative integer pixel coordinates"
    ):
        load_track2p_subject(
            subject_dir,
            input_format="suite2p",
            plane_name="plane0",
            load_traces=False,
            load_spike_traces=False,
        )


def test_load_track2p_subject_rejects_out_of_bounds_suite2p_coordinates(tmp_path):
    subject_dir = tmp_path / "subject"
    plane_dir = subject_dir / "2024-01-01_session" / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True)

    stat = np.asarray(
        [
            {
                "ypix": np.asarray([1, 2], dtype=int),
                "xpix": np.asarray([1, 5], dtype=int),
                "lam": np.asarray([1.0, 1.0], dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "ops.npy", {"Ly": 4, "Lx": 5})

    with pytest.raises(ValueError, match="within image bounds"):
        load_track2p_subject(
            subject_dir,
            input_format="suite2p",
            plane_name="plane0",
            load_traces=False,
            load_spike_traces=False,
        )


def test_load_track2p_subject_rejects_multidimensional_suite2p_coordinates(tmp_path):
    subject_dir = tmp_path / "subject"
    plane_dir = subject_dir / "2024-01-01_session" / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True)

    stat = np.asarray(
        [
            {
                "ypix": np.asarray([[1, 2]], dtype=int),
                "xpix": np.asarray([[1, 2]], dtype=int),
                "lam": np.asarray([[1.0, 1.0]], dtype=float),
                "overlap": np.asarray([[False, False]], dtype=bool),
            }
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "ops.npy", {"Ly": 4, "Lx": 5})

    with pytest.raises(ValueError, match="one-dimensional pixel-coordinate array"):
        load_track2p_subject(
            subject_dir,
            input_format="suite2p",
            plane_name="plane0",
            load_traces=False,
            load_spike_traces=False,
        )


@pytest.mark.parametrize(
    ("ypix", "xpix", "message"),
    [
        (np.asarray([-1, 0], dtype=int), np.asarray([0, 0], dtype=int), "non-negative"),
        (np.asarray([0], dtype=int), np.asarray([4], dtype=int), "within image bounds"),
    ],
)
def test_load_suite2p_plane_rejects_invalid_stat_pixel_bounds(
    tmp_path, ypix, xpix, message
):
    stat = np.asarray(
        [
            {
                "ypix": np.asarray(ypix),
                "xpix": np.asarray(xpix),
                "lam": np.ones(len(ypix), dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)
    np.save(tmp_path / "ops.npy", {"Ly": 4, "Lx": 4})

    with pytest.raises(ValueError, match=message):
        load_suite2p_plane(
            tmp_path,
            load_traces=False,
            load_spike_traces=False,
        )
