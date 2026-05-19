from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from bayescatrack import load_suite2p_plane


def _minimal_suite2p_stat() -> np.ndarray:
    stat = np.empty(2, dtype=object)
    stat[0] = {
        "ypix": np.array([0, 0], dtype=int),
        "xpix": np.array([0, 1], dtype=int),
    }
    stat[1] = {
        "ypix": np.array([1, 1], dtype=int),
        "xpix": np.array([0, 1], dtype=int),
    }
    return stat


def _write_minimal_suite2p_plane(plane_dir: Path) -> None:
    plane_dir.mkdir(parents=True)
    np.save(plane_dir / "stat.npy", _minimal_suite2p_stat())
    np.save(
        plane_dir / "iscell.npy",
        np.array([[1.0, 0.95], [1.0, 0.90]], dtype=float),
    )
    np.save(plane_dir / "F.npy", np.ones((2, 3), dtype=float))
    np.save(plane_dir / "spks.npy", np.zeros((2, 3), dtype=float))
    np.save(plane_dir / "Fneu.npy", np.full((2, 3), 0.5, dtype=float))


def test_load_suite2p_plane_rejects_iscell_roi_count_mismatch(
    tmp_path: Path,
) -> None:
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir)
    np.save(plane_dir / "iscell.npy", np.array([[1.0, 0.95]], dtype=float))

    with pytest.raises(
        ValueError,
        match=r"iscell\.npy has 1 ROI rows, but stat\.npy has 2",
    ):
        load_suite2p_plane(plane_dir)


@pytest.mark.parametrize(
    ("filename", "load_kwargs"),
    [
        ("F.npy", {}),
        ("spks.npy", {}),
        ("Fneu.npy", {"load_neuropil_traces": True}),
    ],
)
def test_load_suite2p_plane_rejects_trace_roi_count_mismatch(
    tmp_path: Path,
    filename: str,
    load_kwargs: dict[str, bool],
) -> None:
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir)
    np.save(plane_dir / filename, np.ones((1, 3), dtype=float))

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(filename)} has 1 ROI rows, but stat\.npy has 2",
    ):
        load_suite2p_plane(plane_dir, **load_kwargs)
