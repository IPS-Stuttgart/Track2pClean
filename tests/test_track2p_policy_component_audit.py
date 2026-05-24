from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    apply_weakest_bridge_splits,
    component_audit_rows,
    edge_risk_score,
    split_track_at_bridge,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
)


@dataclass(frozen=True)
class _Plane:
    roi_indices: np.ndarray

    @property
    def n_rois(self) -> int:
        return int(self.roi_indices.size)


@dataclass(frozen=True)
class _Session:
    roi_indices: tuple[int, ...]

    @property
    def plane_data(self) -> _Plane:
        return _Plane(np.asarray(self.roi_indices, dtype=int))


def test_edge_risk_score_prefers_weak_bridge() -> None:
    weak = _diagnostic(
        threshold_margin=0.01,
        row_margin=0.02,
        column_margin=0.03,
        centroid_distance=8.0,
        area_ratio=0.20,
    )
    strong = _diagnostic(
        threshold_margin=0.30,
        row_margin=0.50,
        column_margin=0.50,
        centroid_distance=2.0,
        area_ratio=0.90,
    )

    assert edge_risk_score(weak) > 3.0
    assert edge_risk_score(strong) == 0.0


def test_split_track_at_bridge_returns_left_and_right_fragments() -> None:
    left, right = split_track_at_bridge(np.asarray([10, 20, 30, 40]), 1)

    np.testing.assert_array_equal(left, [10, 20, -1, -1])
    np.testing.assert_array_equal(right, [-1, -1, 30, 40])


def test_component_audit_marks_weakest_bridge_and_split_application() -> None:
    sessions = [_Session((10,)), _Session((20,)), _Session((30,)), _Session((40,))]
    predicted = np.asarray([[10, 20, 30, 40]], dtype=int)
    reference = np.asarray([[10, 20, 30, 40]], dtype=int)
    diagnostics = (
        _diagnostic(
            session_index=0,
            threshold_margin=0.30,
            row_margin=0.50,
            column_margin=0.50,
            centroid_distance=2.0,
            area_ratio=0.90,
        ),
        _diagnostic(
            session_index=1,
            threshold_margin=0.01,
            row_margin=0.02,
            column_margin=0.03,
            centroid_distance=8.0,
            area_ratio=0.20,
        ),
        _diagnostic(
            session_index=2,
            threshold_margin=0.30,
            row_margin=0.50,
            column_margin=0.50,
            centroid_distance=2.0,
            area_ratio=0.90,
        ),
    )

    rows = component_audit_rows(
        predicted,
        reference,
        sessions=sessions,  # type: ignore[arg-type]
        diagnostics=diagnostics,
        config=ComponentCleanupConfig(split_risk_threshold=1.0),
    )

    assert rows[0]["would_split_at_weakest_edge"] == 1
    assert rows[0]["applied_split"] == 0
    assert rows[0]["weakest_bridge_session_a"] == 1
    assert rows[0]["complete_track_status_against_gt"] == "true_positive"

    no_apply = apply_weakest_bridge_splits(predicted, rows)
    applied_rows = [{**rows[0], "applied_split": 1}]
    cleaned = apply_weakest_bridge_splits(predicted, applied_rows)

    np.testing.assert_array_equal(no_apply, predicted)
    np.testing.assert_array_equal(cleaned, [[10, 20, -1, -1], [-1, -1, 30, 40]])


def _diagnostic(
    *,
    session_index: int = 0,
    threshold_margin: float,
    row_margin: float,
    column_margin: float,
    centroid_distance: float,
    area_ratio: float,
) -> Track2pPolicyLinkDiagnostic:
    return Track2pPolicyLinkDiagnostic(
        session_index=session_index,
        local_roi_a=0,
        local_roi_b=0,
        assigned_iou=0.50,
        threshold=0.40,
        threshold_margin=threshold_margin,
        row_margin=row_margin,
        column_margin=column_margin,
        centroid_distance=centroid_distance,
        area_ratio=area_ratio,
        pruned=False,
        prune_reason="kept",
    )
