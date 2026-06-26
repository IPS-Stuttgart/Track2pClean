from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import load_suite2p_plane


def _write_plane(plane_dir, iscell: np.ndarray) -> None:
    plane_dir.mkdir(parents=True, exist_ok=True)
    stat = np.asarray([
        {"ypix": np.asarray([0]), "xpix": np.asarray([0]), "lam": np.asarray([1.0])}
    ], dtype=object)
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "iscell.npy", iscell)


def test_load_suite2p_plane_rejects_out_of_range_iscell_values(tmp_path) -> None:
    _write_plane(tmp_path, np.asarray([[1.0, 1.1]], dtype=float))

    with pytest.raises(ValueError, match="iscell probabilities"):
        load_suite2p_plane(tmp_path, load_traces=False, load_spike_traces=False)
