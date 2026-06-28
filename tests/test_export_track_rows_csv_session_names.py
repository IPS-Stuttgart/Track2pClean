from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.matching import export_track_rows_csv
from tests import _support  # noqa: F401


def test_export_track_rows_csv_rejects_bare_session_names_string(tmp_path):
    with pytest.raises(ValueError, match="not a bare string"):
        export_track_rows_csv(
            tmp_path / "tracks.csv",
            "s0",
            np.array([[7, -1]], dtype=int),
        )


def test_export_track_rows_csv_rejects_bare_session_names_bytes(tmp_path):
    with pytest.raises(ValueError, match="not a bare string"):
        export_track_rows_csv(
            tmp_path / "tracks.csv",
            b"s0",
            np.array([[7, -1]], dtype=int),
        )


def test_export_track_rows_csv_rejects_duplicate_session_names(tmp_path):
    with pytest.raises(ValueError, match="unique session names"):
        export_track_rows_csv(
            tmp_path / "tracks.csv",
            ["session0", "session0"],
            np.array([[7, -1]], dtype=int),
        )


def test_export_track_rows_csv_accepts_unique_session_names(tmp_path):
    output_path = export_track_rows_csv(
        tmp_path / "tracks.csv",
        ["session0", "session1"],
        np.array([[7, -1]], dtype=int),
    )

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "track_id,session0,session1",
        "0,7,-1",
    ]
