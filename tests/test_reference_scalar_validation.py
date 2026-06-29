from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.reference import Track2pReference, score_pairwise_matches


def test_reference_rejects_out_of_range_integer_observations_as_value_error():
    huge_roi = 10**1000

    with pytest.raises(ValueError, match="ROI index must be integer-like"):
        score_pairwise_matches(
            np.array([[huge_roi, 0]], dtype=object),
            np.array([[1, 0]], dtype=object),
        )


def test_reference_rejects_out_of_range_roi_count_as_value_error():
    reference = Track2pReference(
        session_names=("day0",),
        suite2p_indices=np.array([[0]], dtype=object),
    )

    with pytest.raises(
        ValueError, match="n_rois_per_session must be an integer scalar"
    ):
        reference.to_session_track_labels(n_rois_per_session=[10**1000])
