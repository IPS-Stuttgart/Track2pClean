from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.reference import score_complete_tracks


def test_score_complete_tracks_rejects_fractional_float_roi_indices():
    with pytest.raises(ValueError, match="integer-like"):
        score_complete_tracks([[1.7, 2]], [[1, 2]])


def test_score_complete_tracks_rejects_fractional_string_roi_indices():
    with pytest.raises(ValueError, match="integer-like"):
        score_complete_tracks([["1.7", 2]], [[1, 2]])


def test_score_complete_tracks_rejects_boolean_roi_indices():
    with pytest.raises(ValueError, match="boolean"):
        score_complete_tracks([[True, 2]], [[1, 2]])


def test_score_complete_tracks_rejects_numpy_boolean_roi_indices():
    with pytest.raises(ValueError, match="boolean"):
        score_complete_tracks([[np.bool_(False), 2]], [[0, 2]])


def test_score_complete_tracks_accepts_integer_like_string_roi_indices():
    scores = score_complete_tracks([["1.0", 2]], [[1, 2]])

    assert scores["complete_tracks_score"] == 1.0
