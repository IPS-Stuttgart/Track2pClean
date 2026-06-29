from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from bayescatrack.reference import Track2pReference, score_complete_tracks  # noqa: E402


def test_reference_preserves_large_textual_decimal_roi_indices():
    first_roi = 2**53 + 1
    second_roi = first_roi + 2

    reference = Track2pReference(
        session_names=("day0", "day1"),
        suite2p_indices=np.array([[f"{first_roi}.0", str(second_roi)]], dtype=object),
    )

    npt.assert_array_equal(
        reference.suite2p_indices,
        np.array([[first_roi, second_roi]], dtype=object),
    )
    npt.assert_array_equal(reference.pairwise_matches(0, 1), np.array([[first_roi, second_roi]]))


def test_score_complete_tracks_preserves_large_textual_decimal_indices():
    first_roi = 2**53 + 1
    second_roi = first_roi + 2

    scores = score_complete_tracks(
        np.array([[f"{first_roi}.0", str(second_roi)]], dtype=object),
        np.array([[first_roi, second_roi]], dtype=object),
    )

    assert scores["T_rc"] == 1
    assert scores["T_c"] == 1
    assert scores["T_gt"] == 1
    assert scores["ct"] == 1.0
