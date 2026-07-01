from pathlib import Path

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData, Track2pSession
from bayescatrack.association.teacher_priors import teacher_edge_masks_from_track_matrix


def _session(name: str, n_rois: int = 3) -> Track2pSession:
    roi_masks = np.zeros((n_rois, 2, 2), dtype=bool)
    for roi_index in range(n_rois):
        roi_masks[roi_index, roi_index // 2, roi_index % 2] = True
    return Track2pSession(
        session_dir=Path(name),
        session_name=name,
        session_date=None,
        plane_data=CalciumPlaneData(roi_masks=roi_masks),
    )


@pytest.mark.parametrize("bad_token", [memoryview(b"1"), bytearray(b"1")])
def test_teacher_prior_treats_binary_buffer_roi_cells_as_missing(bad_token):
    sessions = (_session("s0"), _session("s1"))

    masks = teacher_edge_masks_from_track_matrix(
        [[bad_token, 2], [1, bad_token]],
        sessions,
        session_edges=[(0, 1)],
    )

    assert not np.any(masks[(0, 1)])
