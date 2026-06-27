from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import load_track2p_subject


def _write_invalid_suite2p_plane(plane_dir) -> None:
    plane_dir.mkdir(parents=True)

    stat = np.asarray(
        [
            {
                "ypix": np.asarray([0, 1], dtype=int),
                "xpix": np.asarray([0, 1], dtype=int),
                "lam": np.asarray([1.0, 1.0], dtype=float),
                "overlap": np.asarray([False], dtype=bool),
            }
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)


def _write_valid_raw_npy_plane(plane_dir) -> None:
    plane_dir.mkdir(parents=True)

    roi_masks = np.zeros((1, 2, 2), dtype=bool)
    roi_masks[0, 0, 0] = True

    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(plane_dir / "F.npy", np.ones((1, 3), dtype=float))
    np.save(plane_dir / "fov.npy", roi_masks[0].astype(float))


def test_auto_loader_refuses_raw_fallback_when_suite2p_validation_fails(tmp_path):
    session_dir = tmp_path / "2024-01-01"
    _write_invalid_suite2p_plane(session_dir / "suite2p" / "plane0")
    _write_valid_raw_npy_plane(session_dir / "data_npy" / "plane0")

    with pytest.raises(RuntimeError, match="refusing auto fallback"):
        load_track2p_subject(
            tmp_path,
            input_format="auto",
            load_traces=False,
            load_spike_traces=False,
        )
