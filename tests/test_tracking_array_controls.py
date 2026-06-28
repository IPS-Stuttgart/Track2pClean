import bayescatrack.tracking as tracking
import numpy as np
import pytest


def test_seed_roi_list_rejects_zero_dimensional_array_element():
    rows = np.array([[0, 10], [1, 11]], dtype=int)

    with pytest.raises(ValueError, match="start_roi_indices"):
        tracking._restrict_track_rows_to_start_rois(
            rows,
            start_roi_indices=[np.array(0, dtype=np.int64)],
            start_session_index=0,
            fill_value=-1,
        )


def test_start_session_rejects_zero_dimensional_array_scalar():
    rows = np.array([[0, 10], [1, 11]], dtype=int)

    with pytest.raises(ValueError, match="start_session_index"):
        tracking._restrict_track_rows_to_start_rois(
            rows,
            start_roi_indices=[0, 1],
            start_session_index=np.array(0, dtype=np.int64),
            fill_value=-1,
        )
