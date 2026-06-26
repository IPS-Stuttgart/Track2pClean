from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.analysis.growth import affine_growth_summaries, radial_displacement_rows
from bayescatrack.core.bridge import CalciumPlaneData, Track2pSession


def _plane_from_points(points_xy, roi_indices):
    width = max(int(point[0]) for point in points_xy) + 1
    height = max(int(point[1]) for point in points_xy) + 1
    masks = np.zeros((len(points_xy), height, width), dtype=bool)
    for roi_index, (x_coord, y_coord) in enumerate(points_xy):
        masks[roi_index, int(y_coord), int(x_coord)] = True
    return CalciumPlaneData(
        roi_masks=masks,
        roi_indices=np.asarray(roi_indices, dtype=int),
        source="test",
        plane_name="plane0",
    )


def _session(name, points_xy, roi_indices):
    return Track2pSession(
        session_dir=None,
        session_name=name,
        session_date=None,
        plane_data=_plane_from_points(points_xy, roi_indices),
    )


@pytest.mark.parametrize(
    "tracks",
    [
        np.asarray([[999, 20]], dtype=object),
        np.asarray([[10, 999]], dtype=object),
    ],
)
def test_growth_rejects_unknown_positive_roi_ids(tracks):
    sessions = [
        _session("s0", [(3, 2)], [10]),
        _session("s1", [(4, 2)], [20]),
    ]

    with pytest.raises(ValueError, match="not present in the loaded session"):
        radial_displacement_rows(sessions, tracks, center=(2, 2))

    with pytest.raises(ValueError, match="not present in the loaded session"):
        affine_growth_summaries(sessions, tracks)
