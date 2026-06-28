from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import load_suite2p_plane


class _OverflowingFloat:
    def __float__(self) -> float:
        raise OverflowError("overflow while converting to float")


def _write_minimal_suite2p_plane(plane_dir):
    plane_dir.mkdir(parents=True, exist_ok=True)
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([0], dtype=int),
                "xpix": np.asarray([0], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)


def test_load_suite2p_plane_rejects_overflowing_cell_probability_threshold(tmp_path):
    _write_minimal_suite2p_plane(tmp_path)

    with pytest.raises(ValueError, match="cell_probability_threshold"):
        load_suite2p_plane(
            tmp_path,
            load_traces=False,
            load_spike_traces=False,
            cell_probability_threshold=_OverflowingFloat(),
        )
