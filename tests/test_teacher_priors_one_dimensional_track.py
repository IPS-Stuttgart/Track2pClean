from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bayescatrack.association.teacher_priors import (
    apply_teacher_edge_priors,
    teacher_edge_masks_from_track_matrix,
)
from bayescatrack.core.bridge import CalciumPlaneData, Track2pSession


def _session(name: str, roi_indices: list[int]) -> Track2pSession:
    roi_masks = np.zeros((len(roi_indices), 4, 4), dtype=bool)
    for position in range(len(roi_indices)):
        roi_masks[position, position, position] = True
    return Track2pSession(
        session_dir=Path(name),
        session_name=name,
        session_date=None,
        plane_data=CalciumPlaneData(
            roi_masks=roi_masks,
            roi_indices=np.asarray(roi_indices, dtype=int),
            source="teacher-prior-test",
            plane_name="plane0",
        ),
    )


def _sessions() -> tuple[Track2pSession, Track2pSession, Track2pSession]:
    return (
        _session("s0", [10, 11]),
        _session("s1", [20, 21]),
        _session("s2", [30, 31]),
    )


def test_one_dimensional_teacher_track_matrix_is_single_track_row() -> None:
    sessions = _sessions()

    masks = teacher_edge_masks_from_track_matrix(
        [10, 21, 31],
        sessions,
        session_edges=((0, 1), (1, 2), (0, 2)),
    )

    assert masks[(0, 1)].shape == (2, 2)
    assert masks[(1, 2)].shape == (2, 2)
    assert masks[(0, 2)].shape == (2, 2)

    assert masks[(0, 1)][0, 1]
    assert masks[(1, 2)][1, 1]
    assert masks[(0, 2)][0, 1]

    assert int(np.sum(masks[(0, 1)])) == 1
    assert int(np.sum(masks[(1, 2)])) == 1
    assert int(np.sum(masks[(0, 2)])) == 1


@pytest.mark.parametrize(
    "bad_session_edges",
    [
        "01",
        ("01",),
        ((False, True),),
        ((0.5, 1),),
        (("0", "1"),),
        ((0, 0),),
        ((0, 3),),
        ((0, 1, 2),),
        ((0, 1), (0, 1)),
    ],
)
def test_teacher_edge_masks_reject_malformed_session_edges(
    bad_session_edges: object,
) -> None:
    with pytest.raises(ValueError, match="session_edges"):
        teacher_edge_masks_from_track_matrix(
            [10, 21, 31],
            _sessions(),
            session_edges=bad_session_edges,  # type: ignore[arg-type]
        )


def test_apply_teacher_edge_priors_rejects_invalid_pairwise_cost_edges() -> None:
    with pytest.raises(ValueError, match="session_edges"):
        apply_teacher_edge_priors(
            {(False, True): np.zeros((2, 2), dtype=float)},  # type: ignore[dict-item]
            _sessions(),
            teacher_track_matrix=[10, 21, 31],
        )
