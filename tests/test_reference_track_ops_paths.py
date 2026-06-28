from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from bayescatrack.reference import load_track2p_reference  # noqa: E402


def test_load_track2p_reference_uses_session_parent_for_plane_level_paths(
    tmp_path: Path,
):
    track2p_dir = tmp_path / "track2p"
    track2p_dir.mkdir()

    track_ops = {
        "all_ds_path": np.array(
            [
                str(tmp_path / "subject" / "2024-05-01_a" / "suite2p" / "plane0"),
                str(tmp_path / "subject" / "2024-05-02_a" / "data_npy" / "plane0"),
            ],
            dtype=object,
        )
    }
    np.save(track2p_dir / "track_ops.npy", track_ops, allow_pickle=True)
    np.save(
        track2p_dir / "plane0_suite2p_indices.npy",
        np.array([[0, 1]], dtype=object),
        allow_pickle=True,
    )

    reference = load_track2p_reference(track2p_dir, plane_name="plane0")

    assert reference.session_names == ("2024-05-01_a", "2024-05-02_a")
