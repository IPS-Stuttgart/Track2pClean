import numpy as np
import pytest
from bayescatrack.matching import export_track_rows_csv


class _BadIndex:
    def __init__(self, exc_type: type[Exception]) -> None:
        self.exc_type = exc_type

    def __index__(self) -> int:
        raise self.exc_type("bad index")


def test_export_track_rows_csv_respects_numpy_boolean_false(tmp_path):
    output_path = tmp_path / "tracks.csv"

    export_track_rows_csv(
        output_path,
        ("s1", "s2"),
        np.array([[10, -1]]),
        include_track_id=np.bool_(False),
    )

    assert output_path.read_text(encoding="utf-8").splitlines() == ["s1,s2", "10,-1"]


@pytest.mark.parametrize(
    "malformed_include_track_id",
    ["false", "true", 0, 1, None, np.array([True])],
)
def test_export_track_rows_csv_rejects_malformed_include_track_id(
    tmp_path,
    malformed_include_track_id,
):
    with pytest.raises(ValueError, match="include_track_id must be a boolean"):
        export_track_rows_csv(
            tmp_path / "tracks.csv",
            ("s1", "s2"),
            np.array([[10, -1]]),
            include_track_id=malformed_include_track_id,
        )


@pytest.mark.parametrize(
    "malformed_track_rows",
    [
        [[True, -1]],
        [[np.bool_(False), -1]],
        [[10.5, -1]],
        [[np.nan, -1]],
        [[np.inf, -1]],
        [["10", -1]],
        [[None, -1]],
    ],
)
def test_export_track_rows_csv_rejects_malformed_positional_track_rows(
    tmp_path,
    malformed_track_rows,
):
    with pytest.raises(ValueError, match="track_rows must contain integer"):
        export_track_rows_csv(
            tmp_path / "tracks.csv",
            ("s1", "s2"),
            malformed_track_rows,
        )


@pytest.mark.parametrize("exc_type", [ValueError, OverflowError])
def test_export_track_rows_csv_normalizes_bad_index_protocol_errors(tmp_path, exc_type):
    with pytest.raises(ValueError, match="track_rows must contain integer"):
        export_track_rows_csv(
            tmp_path / "tracks.csv",
            ("s1", "s2"),
            [[_BadIndex(exc_type), -1]],
        )


def test_export_track_rows_csv_rejects_malformed_keyword_track_rows(tmp_path):
    with pytest.raises(ValueError, match="track_rows must contain integer"):
        export_track_rows_csv(
            tmp_path / "tracks.csv",
            ("s1", "s2"),
            track_rows=[["10", -1]],
        )
