from pathlib import Path

import numpy as np
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


def test_teacher_prior_accepts_scalar_numpy_roi_index_cells():
    sessions = (_session("s0"), _session("s1"))

    masks = teacher_edge_masks_from_track_matrix(
        [[np.asarray(1), np.asarray(2)]],
        sessions,
        session_edges=[(0, 1)],
    )

    mask = masks[(0, 1)]
    assert mask[1, 2]
    assert int(np.count_nonzero(mask)) == 1


def test_teacher_prior_treats_vector_roi_index_cells_as_missing():
    sessions = (_session("s0"), _session("s1"))

    masks = teacher_edge_masks_from_track_matrix(
        [[np.asarray([1]), 2]],
        sessions,
        session_edges=[(0, 1)],
    )

    assert not np.any(masks[(0, 1)])
