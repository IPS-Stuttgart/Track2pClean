from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.fixed_precision import (
    score_complete_tracks_at_fixed_precision,
)


def test_fixed_precision_counts_duplicate_complete_tracks_as_false_positives() -> None:
    reference = np.array([[0, 10], [1, 11]], dtype=object)
    predicted = np.array([[0, 10], [0, 10], [1, 11]], dtype=object)

    scores = score_complete_tracks_at_fixed_precision(
        predicted,
        reference,
        target_precisions=(0.9, 0.6),
        track_scores=(0.9, 0.8, 0.7),
    )

    assert scores["complete_tracks_at_fixed_precision_0_9"] == 1
    assert scores["complete_track_predictions_at_fixed_precision_0_9"] == 1
    assert scores["complete_track_precision_at_fixed_precision_0_9"] == pytest.approx(
        1.0
    )
    assert scores["complete_tracks_at_fixed_precision_0_6"] == 2
    assert scores["complete_track_predictions_at_fixed_precision_0_6"] == 3
    assert scores["complete_track_precision_at_fixed_precision_0_6"] == pytest.approx(
        2.0 / 3.0
    )
