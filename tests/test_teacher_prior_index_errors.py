from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData, Track2pSession
from bayescatrack.association.teacher_priors import teacher_edge_masks_from_track_matrix


class _OverflowingIndex:
    def __index__(self) -> int:
        raise OverflowError("bad index")


def _session(name: str) -> Track2pSession:
    roi_masks = np.ones((1, 2, 2), dtype=bool)
    return Track2pSession(
        session_dir=Path(name),
        session_name=name,
        session_date=None,
        plane_data=CalciumPlaneData(roi_masks=roi_masks),
    )


def test_teacher_prior_rejects_overflowing_session_edge_index() -> None:
    sessions = (_session("s0"), _session("s1"))

    with pytest.raises(ValueError, match="session_edges\\[0\\] source"):
        teacher_edge_masks_from_track_matrix(
            [[0, 0]],
            sessions,
            session_edges=[(_OverflowingIndex(), 1)],
        )
