import numpy as np
import pytest
from bayescatrack.matching import build_track_rows_from_matches


@pytest.mark.parametrize(
    ("start_roi_indices", "match"),
    [
        ([True], "start_roi_indices contains boolean ROI index"),
        ([np.bool_(False)], "start_roi_indices contains boolean ROI index"),
        ([-1], "start_roi_indices must contain non-negative ROI indices"),
        ([1.5], "start_roi_indices must contain integer ROI indices"),
        ([1, 1], "start_roi_indices must contain unique ROI indices"),
    ],
)
def test_build_track_rows_from_matches_rejects_invalid_start_roi_indices(
    start_roi_indices,
    match,
):
    with pytest.raises(ValueError, match=match):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[1, 2]], dtype=object)],
            start_roi_indices=start_roi_indices,
        )


@pytest.mark.parametrize(
    ("pairs", "match"),
    [
        ([[True, 2]], "reference_roi_indices contains boolean ROI index"),
        ([[1, np.bool_(False)]], "measurement_roi_indices contains boolean ROI index"),
        ([[-1, 2]], "reference_roi_indices must contain non-negative ROI indices"),
        ([[1, -2]], "measurement_roi_indices must contain non-negative ROI indices"),
        ([[1.5, 2]], "reference_roi_indices must contain integer ROI indices"),
        ([[1, np.nan]], "measurement_roi_indices must contain integer ROI indices"),
    ],
)
def test_build_track_rows_from_matches_rejects_invalid_match_roi_indices(
    pairs,
    match,
):
    with pytest.raises(ValueError, match=match):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array(pairs, dtype=object)],
            start_roi_indices=[1],
        )
