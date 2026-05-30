from __future__ import annotations

import numpy as np

from bayescatrack.experiments import (
    track2p_policy_confidence_ordered_strict_gap_cleanup as mod,
)
from bayescatrack.experiments.track2p_policy_gap_edge_audit import GapEdgeFeature
from bayescatrack.experiments.track2p_policy_strict_gated_gap_cleanup import (
    StrictGapCandidate,
    StrictGapGateConfig,
)


def _feat(
    *,
    registered_iou: float,
    threshold_margin: float,
    area_ratio: float,
    row_margin: float,
    column_margin: float,
) -> GapEdgeFeature:
    return GapEdgeFeature(
        registered_iou=registered_iou,
        threshold=0.25,
        threshold_margin=threshold_margin,
        centroid_distance=4.0,
        area_ratio=area_ratio,
        row_rank=1,
        column_rank=1,
        row_margin=row_margin,
        column_margin=column_margin,
    )


def test_confidence_ordered_merge_uses_stronger_accepted_candidate(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_cell_probability", lambda *_args: 0.95)
    base = np.asarray([[1, -1, -1], [2, -1, -1]], dtype=int)
    proposed = np.asarray([[1, -1, 3], [2, -1, 3]], dtype=int)
    first = StrictGapCandidate(
        edge=(0, 2, 1, 3), candidate_track_id=0, accepted=True, reason="accepted"
    )
    second = StrictGapCandidate(
        edge=(0, 2, 2, 3), candidate_track_id=1, accepted=True, reason="accepted"
    )
    features = {
        first.edge: _feat(
            registered_iou=0.54,
            threshold_margin=0.21,
            area_ratio=0.91,
            row_margin=0.01,
            column_margin=0.01,
        ),
        second.edge: _feat(
            registered_iou=0.25,
            threshold_margin=0.45,
            area_ratio=0.99,
            row_margin=0.20,
            column_margin=0.20,
        ),
    }

    order = mod.confidence_ordered_candidate_indices(
        (first, second),
        sessions=(None, None, None),
        feature_index=features,
        gate_config=StrictGapGateConfig(),
    )
    output, applied = mod.apply_confidence_ordered_strict_gated_gap_candidates_with_report(
        base,
        proposed,
        (first, second),
        sessions=(None, None, None),
        feature_index=features,
        gate_config=StrictGapGateConfig(),
    )

    assert order == (1, 0)
    assert applied == frozenset({1})
    np.testing.assert_array_equal(output, [[1, -1, -1], [2, -1, 3]])
