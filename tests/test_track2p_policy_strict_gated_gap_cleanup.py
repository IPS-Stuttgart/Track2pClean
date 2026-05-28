from __future__ import annotations

import numpy as np
from bayescatrack.experiments import (
    track2p_policy_strict_gated_gap_cleanup as strict_gap,
)
from bayescatrack.experiments.track2p_policy_gap_edge_audit import GapEdgeFeature


def _feature(
    *,
    registered_iou: float = 0.50,
    threshold_margin: float = 0.25,
    area_ratio: float = 0.95,
    row_margin: float = 0.0,
    column_margin: float = 0.0,
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


def test_strict_gated_gap_parser_defaults_match_audit_gate() -> None:
    args = strict_gap.build_arg_parser().parse_args(["--data", "track2p-root"])

    assert args.max_gap == 2
    assert args.gate_gap_length == 2
    assert args.gate_min_area_ratio == 0.90
    assert args.gate_min_cell_probability == 0.80
    assert args.gate_max_registered_iou == 0.55
    assert args.gate_min_row_margin == 0.0
    assert args.gate_min_column_margin == 0.0
    assert args.gate_min_threshold_margin == 0.20


def test_strict_gap_gate_accepts_audit_selected_feature(monkeypatch) -> None:
    monkeypatch.setattr(strict_gap, "_cell_probability", lambda _session, _roi: 0.85)

    accepted, reason = strict_gap.strict_gap_gate_decision(
        (0, 2, 853, 554),
        _feature(),
        sessions=(object(), object(), object()),
        gate_config=strict_gap.StrictGapGateConfig(),
    )

    assert accepted is True
    assert reason == "accepted"


def test_strict_gap_gate_rejects_weak_feature(monkeypatch) -> None:
    monkeypatch.setattr(strict_gap, "_cell_probability", lambda _session, _roi: 0.85)

    accepted, reason = strict_gap.strict_gap_gate_decision(
        (0, 2, 853, 554),
        _feature(registered_iou=0.60, area_ratio=0.80),
        sessions=(object(), object(), object()),
        gate_config=strict_gap.StrictGapGateConfig(),
    )

    assert accepted is False
    assert "registered-iou" in reason
    assert "area-ratio" in reason


def test_apply_strict_gated_gap_candidates_merges_suffix_observation() -> None:
    base = np.asarray([[853, -1, -1]], dtype=int)
    candidate_tracks = np.asarray([[853, -1, 554]], dtype=int)
    candidates = (
        strict_gap.StrictGapCandidate(
            edge=(0, 2, 853, 554),
            candidate_track_id=0,
            accepted=True,
            reason="accepted",
        ),
    )

    output = strict_gap.apply_strict_gated_gap_candidates(
        base, candidate_tracks, candidates
    )

    np.testing.assert_array_equal(output, [[853, -1, 554]])


def test_apply_strict_gated_gap_candidates_rejects_duplicate_observation() -> None:
    base = np.asarray([[853, -1, -1], [999, -1, 554]], dtype=int)
    candidate_tracks = np.asarray([[853, -1, 554]], dtype=int)
    candidates = (
        strict_gap.StrictGapCandidate(
            edge=(0, 2, 853, 554),
            candidate_track_id=0,
            accepted=True,
            reason="accepted",
        ),
    )

    output = strict_gap.apply_strict_gated_gap_candidates(
        base, candidate_tracks, candidates
    )

    np.testing.assert_array_equal(output, base)
