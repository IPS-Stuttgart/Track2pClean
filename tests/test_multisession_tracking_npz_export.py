import numpy as np
from bayescatrack.multisession_tracking import (
    LongitudinalTrackingResult,
    save_tracking_result_npz,
)


def test_save_tracking_result_npz_uses_pickle_free_string_metadata(tmp_path):
    result = LongitudinalTrackingResult(
        tracks=tuple(),
        track_matrix=np.zeros((0, 2), dtype=int),
        track_roi_index_matrix=np.zeros((0, 2), dtype=int),
        session_names=("2024-05-01_a", "undated_session"),
        session_dates=("2024-05-01", None),
        pairwise_bundles=tuple(),
        total_cost=None,
    )

    output_path = tmp_path / "tracks.npz"
    summary = save_tracking_result_npz(result, output_path)

    assert summary["output_path"] == str(output_path)
    with np.load(output_path, allow_pickle=False) as exported:
        assert exported["session_names"].dtype.kind == "U"
        assert exported["session_dates"].dtype.kind == "U"
        assert exported["session_names"].tolist() == [
            "2024-05-01_a",
            "undated_session",
        ]
        assert exported["session_dates"].tolist() == ["2024-05-01", ""]
