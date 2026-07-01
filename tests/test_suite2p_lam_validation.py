from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import load_suite2p_plane


def test_weighted_loader_rejects_invalid_raw_lam_before_duplicate_overwrite(tmp_path):
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([0, 0], dtype=int),
                "xpix": np.asarray([0, 0], dtype=int),
                "lam": np.asarray([-1.0, 1.0], dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)

    with pytest.raises(ValueError, match="lam values"):
        load_suite2p_plane(
            tmp_path,
            weighted_masks=True,
            load_traces=False,
            load_spike_traces=False,
        )
