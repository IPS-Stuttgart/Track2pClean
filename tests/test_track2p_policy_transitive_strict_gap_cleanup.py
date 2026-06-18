from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.experiments.track2p_policy_gap_edge_audit import GapEdgeFeature
from bayescatrack.experiments.track2p_policy_strict_gated_gap_cleanup import (
    StrictGapGateConfig,
)
from bayescatrack.experiments.track2p_policy_transitive_strict_gap_cleanup import (
    apply_transitive_strict_gated_gap_edges_with_report,
)


def test_transitive_gap_cleanup_follows_newly_inserted_source() -> None:
    base_tracks = np.asarray([[10, -1, -1, -1, -1]], dtype=int)
    gate_config = StrictGapGateConfig(
        gap_length=2,
        min_area_ratio=0.90,
        min_cell_probability=0.80,
        max_registered_iou=0.55,
        min_threshold_margin=0.20,
    )
    feature_index = {
        (0, 2, 10, 30): _feature(threshold_margin=0.30),
        (2, 4, 30, 50): _feature(threshold_margin=0.35),
    }

    cleaned, candidates, applied = apply_transitive_strict_gated_gap_edges_with_report(
        base_tracks,
        sessions=(
            _session(10),
            _session(20),
            _session(30),
            _session(40),
            _session(50),
        ),
        feature_index=feature_index,
        gate_config=gate_config,
        seed_rois={10},
    )

    assert cleaned.tolist() == [[10, -1, 30, -1, 50]]
    assert [candidate.edge for candidate in candidates] == [
        (0, 2, 10, 30),
        (2, 4, 30, 50),
    ]
    assert applied == frozenset({0, 1})


@pytest.mark.parametrize(
    "max_rounds",
    [True, False, 0, -1, 1.5, "2", float("nan"), float("inf")],
)
def test_transitive_gap_cleanup_rejects_invalid_max_rounds(
    max_rounds: object,
) -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        apply_transitive_strict_gated_gap_edges_with_report(
            np.asarray([[10, -1, -1]], dtype=int),
            sessions=(_session(10), _session(20), _session(30)),
            feature_index={(0, 2, 10, 30): _feature(threshold_margin=0.30)},
            gate_config=StrictGapGateConfig(),
            seed_rois={10},
            max_rounds=max_rounds,  # type: ignore[arg-type]
        )


def _feature(*, threshold_margin: float) -> GapEdgeFeature:
    return GapEdgeFeature(
        registered_iou=0.50,
        threshold=0.20,
        threshold_margin=threshold_margin,
        centroid_distance=5.0,
        area_ratio=0.95,
        row_rank=1,
        column_rank=1,
        row_margin=0.10,
        column_margin=0.10,
    )


def _session(suite2p_roi: int) -> SimpleNamespace:
    plane_data = SimpleNamespace(
        cell_probabilities=np.asarray([0.95], dtype=float),
        roi_indices=np.asarray([suite2p_roi], dtype=int),
    )
    return SimpleNamespace(
        session_dir=Path("."),
        session_name=f"session-{suite2p_roi}",
        session_date=None,
        plane_data=plane_data,
    )
