from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from bayescatrack.ground_truth_eval import load_track_table_csv


def test_load_long_format_rejects_duplicate_track_session_rows(tmp_path: Path):
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text(
        "track_id,session,roi\n"
        "cell0,day0,1\n"
        "cell0,day0,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate long-format track/session entry"):
        load_track_table_csv(csv_path, session_names=("day0",))


def test_load_long_format_rejects_duplicate_missing_track_session_rows(
    tmp_path: Path,
):
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text(
        "track_id,session,roi\n"
        "cell0,day0,\n"
        "cell0,day0,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate long-format track/session entry"):
        load_track_table_csv(csv_path, session_names=("day0",))


def test_load_long_format_allows_one_row_per_track_and_session(tmp_path: Path):
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text(
        "track_id,session,roi\ncell0,day0,1\ncell0,day1,4\ncell1,day0,2\n",
        encoding="utf-8",
    )

    table = load_track_table_csv(csv_path, session_names=("day0", "day1"))

    assert table.session_names == ("day0", "day1")
    npt.assert_array_equal(table.tracks, np.array([[1, 4], [2, -1]], dtype=int))
