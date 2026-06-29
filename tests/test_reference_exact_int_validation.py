from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from bayescatrack import reference  # noqa: E402
from bayescatrack.reference import Track2pReference, score_complete_tracks  # noqa: E402


def test_reference_preserves_large_textual_decimal_roi_indices():
    first_roi = 2**53 + 1
    second_roi = first_roi + 2

    reference_obj = Track2pReference(
        session_names=("day0", "day1"),
        suite2p_indices=np.array([[f"{first_roi}.0", str(second_roi)]], dtype=object),
    )

    npt.assert_array_equal(
        reference_obj.suite2p_indices,
        np.array([[first_roi, second_roi]], dtype=object),
    )
    npt.assert_array_equal(
        reference_obj.pairwise_matches(0, 1), np.array([[first_roi, second_roi]])
    )


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


def test_strict_reference_integer_scalar_rejects_textual_decimal_nan():
    with pytest.raises(ValueError, match="fill_value must be an integer scalar"):
        reference._parse_integer_scalar(  # pylint: disable=protected-access
            "NaN",
            name="fill_value",
            allow_negative=True,
            allow_string=True,
        )


def test_optional_reference_roi_textual_decimal_nan_stays_missing():
    assert (
        reference._parse_optional_int("NaN") is None
    )  # pylint: disable=protected-access
    assert (
        reference._parse_optional_int("sNaN") is None
    )  # pylint: disable=protected-access
