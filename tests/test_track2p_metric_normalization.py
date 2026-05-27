from __future__ import annotations

import pytest
from bayescatrack.evaluation.track2p_metrics import normalize_track_matrix


def test_track2p_metric_normalization_rejects_fractional_roi_indices():
    with pytest.raises(ValueError, match="ROI observations must be integer-like"):
        normalize_track_matrix([[0, 1.5]])


def test_track2p_metric_normalization_accepts_missing_observations():
    normalized = normalize_track_matrix([[0, "nan", None, -1]])

    assert normalized.shape == (1, 4)
    assert normalized[0, 0] == 0
    assert all(normalized[0, index] < 0 for index in (1, 2, 3))
