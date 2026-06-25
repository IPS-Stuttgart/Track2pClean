from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import load_track2p_subject


def _write_minimal_suite2p_plane(plane_dir):
    plane_dir.mkdir(parents=True)
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
    "flag_name",
    ["load_traces", "load_spike_traces", "load_neuropil_traces"],
)
def test_load_track2p_subject_rejects_non_python_bool_trace_controls(tmp_path, flag_name):
    _write_minimal_suite2p_plane(tmp_path / "2024-01-01" / "suite2p" / "plane0")

    with pytest.raises(ValueError, match=flag_name):
        load_track2p_subject(tmp_path, **{flag_name: np.bool_(False)})
