from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import load_suite2p_plane


def _write_minimal_suite2p_stat(plane_dir) -> None:
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


@pytest.mark.parametrize(
    "iscell",
    [
        np.asarray([[np.nan, 0.9]], dtype=float),
        np.asarray([[1.0, np.inf]], dtype=float),
        np.asarray([[1.0, 1.2]], dtype=float),
        np.asarray([["False", "0.9"]], dtype=object),
    ],
)
def test_suite2p_loader_rejects_invalid_iscell_values(tmp_path, iscell):
    _write_minimal_suite2p_stat(tmp_path)
    np.save(tmp_path / "iscell.npy", iscell)

    with pytest.raises(ValueError, match="iscell.npy .*finite numbers"):
        load_suite2p_plane(tmp_path, load_traces=False, load_spike_traces=False)


@pytest.mark.parametrize("flag_value", [0.2, 0.5, 0.999])
def test_suite2p_loader_rejects_fractional_2d_cell_flags(tmp_path, flag_value):
    _write_minimal_suite2p_stat(tmp_path)
    np.save(tmp_path / "iscell.npy", np.asarray([[flag_value, 0.9]], dtype=float))

    with pytest.raises(ValueError, match="cell-flag column.*binary"):
        load_suite2p_plane(tmp_path, load_traces=False, load_spike_traces=False)
