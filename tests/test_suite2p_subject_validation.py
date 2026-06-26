import numpy as np
import pytest

from bayescatrack import load_track2p_subject


def test_load_track2p_subject_rejects_mismatched_suite2p_lam_shape(tmp_path):
    subject_dir = tmp_path / "jm123"
    plane_dir = subject_dir / "2024-05-01_a" / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True)

    stat = np.empty(1, dtype=object)
    stat[0] = {
        "ypix": np.array([0, 1]),
        "xpix": np.array([0, 1]),
        "lam": np.array([1.0]),
    }
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "ops.npy", {"Ly": 2, "Lx": 2, "meanImg": np.ones((2, 2))})

    with pytest.raises(ValueError, match="ROI 0 lam shape must match"):
        load_track2p_subject(subject_dir, input_format="suite2p", weighted_masks=True)
