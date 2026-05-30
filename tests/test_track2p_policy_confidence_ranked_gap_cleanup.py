from __future__ import annotations

import numpy as np
from bayescatrack.experiments import (
    track2p_policy_confidence_ranked_gap_cleanup as ranked_gap,
)
from bayescatrack.experiments import (
    track2p_policy_strict_gated_gap_cleanup as strict_gap,
)
from bayescatrack.experiments.track2p_policy_gap_edge_audit import GapEdgeFeature


def _feature(margin: float) -> GapEdgeFeature:
    return GapEdgeFeature(
        registered_iou=0.5,
        threshold=0.25,
        threshold_margin=margin,
        centroid_distance=4.0,
        area_ratio=0.95,
        row_rank=1,
        column_rank=1,
        row_margin=0.0,
        column_margin=0.0,
    )


def test_confidence_ranked_gap_edges_resolve_target_conflicts_by_margin(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ranked_gap, "_cell_probability", lambda _session, _roi: 0.85)
    base = np.asarray([[1, -1, -1], [2, -1, -1]], dtype=int)
    candidates = (
        strict_gap.StrictGapCandidate((0, 2, 1, 9), 0, True, "accepted"),
        strict_gap.StrictGapCandidate((0, 2, 2, 9), 1, True, "accepted"),
    )

    output, ranked, applied = (
        ranked_gap.apply_confidence_ranked_strict_gap_edges_with_report(
            base,
            candidates,
            sessions=(object(), object(), object()),
            feature_index={
                (0, 2, 1, 9): _feature(0.25),
                (0, 2, 2, 9): _feature(0.50),
            },
        )
    )

    assert ranked[0].edge == (0, 2, 2, 9)
    assert applied == frozenset({0})
    np.testing.assert_array_equal(output, [[1, -1, -1], [2, -1, 9]])
