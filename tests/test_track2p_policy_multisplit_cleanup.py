from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_multisplit_cleanup import (
    MultiSplitCleanupConfig,
    apply_ranked_bridge_splits,
    plan_ranked_bridge_splits,
    split_track_at_bridges,
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


def test_split_track_at_bridges_returns_all_fragments() -> None:
    fragments = split_track_at_bridges(np.asarray([10, 20, 30, 40, 50, 60]), (1, 3))

    assert len(fragments) == 3
    np.testing.assert_array_equal(fragments[0], [10, 20, -1, -1, -1, -1])
    np.testing.assert_array_equal(fragments[1], [-1, -1, 30, 40, -1, -1])
    np.testing.assert_array_equal(fragments[2], [-1, -1, -1, -1, 50, 60])


def test_plan_ranked_bridge_splits_selects_multiple_guarded_weak_bridges() -> None:
    sessions = [
        _Session((10,)),
        _Session((20,)),
        _Session((30,)),
        _Session((40,)),
        _Session((50,)),
        _Session((60,)),
    ]
    predicted = np.asarray([[10, 20, 30, 40, 50, 60]], dtype=int)
    diagnostics = (
        _diagnostic(session_index=0, threshold_margin=0.30, row_margin=0.50, column_margin=0.50),
        _diagnostic(session_index=1, threshold_margin=0.01, row_margin=0.02, column_margin=0.03),
        _diagnostic(session_index=2, threshold_margin=0.01, row_margin=0.02, column_margin=0.03),
        _diagnostic(session_index=3, threshold_margin=0.01, row_margin=0.02, column_margin=0.03),
        _diagnostic(session_index=4, threshold_margin=0.30, row_margin=0.50, column_margin=0.50),
    )

    split_plan = plan_ranked_bridge_splits(
        predicted,
        sessions=sessions,  # type: ignore[arg-type]
        diagnostics=diagnostics,
        config=MultiSplitCleanupConfig(
            component=ComponentCleanupConfig(split_risk_threshold=1.0),
            max_splits_per_component=3,
        ),
    )

    assert split_plan == {0: (1, 3)}


def test_apply_ranked_bridge_splits_keeps_unplanned_components() -> None:
    predicted = np.asarray(
        [
            [10, 20, 30, 40, 50, 60],
            [11, 21, 31, 41, 51, 61],
        ],
        dtype=int,
    )

    cleaned = apply_ranked_bridge_splits(predicted, {0: (1, 3)})

    np.testing.assert_array_equal(
        cleaned,
        [
            [10, 20, -1, -1, -1, -1],
            [-1, -1, 30, 40, -1, -1],
            [-1, -1, -1, -1, 50, 60],
            [11, 21, 31, 41, 51, 61],
        ],
    )


def test_multisplit_respects_complete_track_guard() -> None:
    sessions = [_Session((10,)), _Session((20,)), _Session((30,)), _Session((40,))]
    predicted = np.asarray([[10, 20, 30, -1]], dtype=int)
    diagnostics = (
        _diagnostic(session_index=0, threshold_margin=0.01, row_margin=0.02, column_margin=0.03),
        _diagnostic(session_index=1, threshold_margin=0.01, row_margin=0.02, column_margin=0.03),
    )

    guarded = plan_ranked_bridge_splits(
        predicted,
        sessions=sessions,  # type: ignore[arg-type]
        diagnostics=diagnostics,
        config=MultiSplitCleanupConfig(
            component=ComponentCleanupConfig(
                split_risk_threshold=1.0,
                min_side_observations=1,
                require_complete_track=True,
            ),
        ),
    )
    unguarded = plan_ranked_bridge_splits(
        predicted,
        sessions=sessions,  # type: ignore[arg-type]
        diagnostics=diagnostics,
        config=MultiSplitCleanupConfig(
            component=ComponentCleanupConfig(
                split_risk_threshold=1.0,
                min_side_observations=1,
                require_complete_track=False,
            ),
        ),
    )

    assert guarded == {}
    assert unguarded == {0: (0, 1)}


def _diagnostic(
    *,
    session_index: int,
    threshold_margin: float,
    row_margin: float,
    column_margin: float,
    centroid_distance: float = 8.0,
    area_ratio: float = 0.20,
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
