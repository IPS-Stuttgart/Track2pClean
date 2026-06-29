from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.reference import Track2pReference, score_pairwise_matches


def test_pair_scoring_preserves_large_non_text_integer_roi_ids() -> None:
    exact_roi = 2**53 + 1
    rounded_neighbor = 2**53

    scores = score_pairwise_matches([[exact_roi, 7]], [[rounded_neighbor, 7]])

    assert scores["true_positives"] == 0
    assert scores["false_positives"] == 1
    assert scores["false_negatives"] == 1


def test_reference_scalar_counts_reject_values_outside_platform_range() -> None:
    reference = Track2pReference(("2024-05-01_a",), [[0]])
    too_large = int(np.iinfo(np.intp).max) + 1

    with pytest.raises(ValueError, match="n_rois_per_session"):
        reference.to_session_track_labels(n_rois_per_session=[too_large])
