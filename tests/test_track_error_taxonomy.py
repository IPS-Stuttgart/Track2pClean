from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.evaluation.track_error_taxonomy import classify_track_errors


@pytest.mark.parametrize("bad_value", [True, np.bool_(False), 1.5, "bad"])
def test_classify_track_errors_rejects_malformed_track_cells(bad_value) -> None:
    with pytest.raises(ValueError, match="predicted_track_matrix"):
        classify_track_errors([[bad_value, 2]], [[1, 2]])


def test_classify_track_errors_keeps_missing_observations_missing() -> None:
    report = classify_track_errors(
        np.asarray([[10, -1], [11, 21]], dtype=int),
        np.asarray([[10, 20], [11, -1]], dtype=int),
    )

    assert {
        (record.kind, record.roi_a, record.roi_b) for record in report.records
    } == {
        ("false_negative", 10, 20),
        ("false_positive", 11, 21),
    }
