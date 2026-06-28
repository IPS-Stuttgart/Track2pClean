from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.core.bridge import _bridge_impl


def test_core_loader_rejects_mismatched_suite2p_overlap_shape(tmp_path):
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
    np.save(tmp_path / "stat.npy", stat)

    with pytest.raises(ValueError, match="overlap shape"):
        _bridge_impl.load_suite2p_plane(
            tmp_path,
            load_traces=False,
            load_spike_traces=False,
        )
