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


def _growth_inputs():
    sessions = [
        _session("s0", [(1, 1), (1, 3), (3, 1), (3, 3)], [0, 1, 2, 3]),
        _session("s1", [(2, 1), (2, 4), (6, 1), (6, 4)], [10, 11, 12, 13]),
    ]
    tracks = np.asarray(
        [
            [0, 10],
            [1, 11],
            [2, 12],
            [3, 13],
        ],
        dtype=object,
    )
    return sessions, tracks


@pytest.mark.parametrize(
    "source_session",
    [True, False, np.bool_(True), np.asarray(True), np.asarray([True], dtype=bool)],
)
def test_growth_radial_rejects_boolean_source_session(source_session):
    sessions, tracks = _growth_inputs()

    with pytest.raises(ValueError, match="boolean|integer-like"):
        radial_displacement_rows(
            sessions,
            tracks,
            source_session=source_session,
            center=(0.0, 0.0),
        )


@pytest.mark.parametrize(
    "target_session",
    [True, False, np.bool_(True), np.asarray(True), np.asarray([True], dtype=bool)],
)
def test_growth_affine_rejects_boolean_target_session(target_session):
    sessions, tracks = _growth_inputs()

    with pytest.raises(ValueError, match="boolean|integer-like"):
        affine_growth_summaries(sessions, tracks, target_sessions=(target_session,))
