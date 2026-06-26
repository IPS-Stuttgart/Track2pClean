from __future__ import annotations

import bayescatrack.evaluation as evaluation
import pytest
from bayescatrack.evaluation.track2p_metrics import (
    normalize_track_matrix as track2p_normalize_track_matrix,
)


def test_evaluation_facade_exposes_track2p_normalizer():
    assert evaluation.normalize_track_matrix is track2p_normalize_track_matrix


def test_evaluation_facade_normalizer_rejects_boolean_roi_indices():
    with pytest.raises(ValueError, match="boolean ROI index"):
        evaluation.normalize_track_matrix([[True, 2]])
