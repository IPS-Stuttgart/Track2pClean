from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from bayescatrack.experiments import track2p_policy_growth_veto_whatif as veto


def test_growth_veto_diagnostic_helpers_ignore_below_zero_local_indices(monkeypatch) -> None:
    sessions = (object(), object())
    diagnostics = (
        _diagnostic(local_roi_a=-1, local_roi_b=0),
        _diagnostic(local_roi_a=0, local_roi_b=-1),
    )
    tracks = np.asarray(
        [
            [20, 11],
            [10, 21],
        ],
        dtype=int,
    )

    def fake_roi_indices(session: object) -> np.ndarray:
        return np.asarray([10, 20]) if session is sessions[0] else np.asarray([11, 21])

    monkeypatch.setattr(veto, "_roi_indices", fake_roi_indices)
    monkeypatch.setattr(veto, "_cell_probability", lambda *args: 0.90)

    anchors = veto._anchor_edges_from_policy_diagnostics(
        sessions,
        diagnostics=diagnostics,
        track2p=tracks,
        component_cleanup=tracks,
        combined=tracks,
        min_registered_iou=0.50,
        min_cell_probability=0.80,
    )
    feature_index = veto._policy_feature_index_from_diagnostics(
        sessions,
        diagnostics,
    )

    assert anchors == {}
    assert feature_index == {}


def _diagnostic(*, local_roi_a: int, local_roi_b: int) -> SimpleNamespace:
    return SimpleNamespace(
        session_index=0,
        local_roi_a=local_roi_a,
        local_roi_b=local_roi_b,
        assigned_iou=0.75,
        centroid_distance=3.5,
        area_ratio=0.91,
        row_margin=0.12,
        column_margin=0.08,
        threshold=0.40,
        threshold_margin=0.35,
    )
