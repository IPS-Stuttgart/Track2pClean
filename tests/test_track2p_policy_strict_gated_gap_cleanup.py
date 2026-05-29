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


def test_delta_gap_candidate_occurrences_returns_only_baseline_delta() -> None:
    base = np.asarray([[1, -1, 3]], dtype=int)
    candidate = np.asarray([[1, -1, 3], [1, -1, 4]], dtype=int)

    occurrences = strict_gap._delta_gap_candidate_occurrences(
        base,
        candidate,
        max_gap=2,
        seed_rois={1},
        seed_session=0,
    )

    assert occurrences == (((0, 2, 1, 4), 1),)


def test_strict_gap_feature_subset_computes_only_requested_pairs(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []

    def fake_roi_indices(session):
        return np.asarray(session, dtype=int)

    def fake_accepted_pair_features(reference_session, moving_session, **kwargs):
        calls.append(
            (
                tuple(int(value) for value in reference_session),
                tuple(int(value) for value in moving_session),
                float(kwargs["iou_distance_threshold"]),
            )
        )
        return {(0, 0): _feature(), (1, 1): _feature(area_ratio=0.50)}

    monkeypatch.setattr(strict_gap, "_roi_indices", fake_roi_indices)
    monkeypatch.setattr(
        strict_gap, "_accepted_pair_features", fake_accepted_pair_features
    )

    output = strict_gap.strict_gap_feature_subset(
        (np.asarray([10, 11]), np.asarray([20]), np.asarray([30, 31])),
        edges={(0, 2, 10, 30)},
        transform_type="affine",
        threshold_method="min",
        iou_distance_threshold=12.0,
    )

    assert calls == [((10, 11), (30, 31), 24.0)]
    assert set(output) == {(0, 2, 10, 30)}


def test_strict_gated_gap_edge_candidates_uses_component_cleanup_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(strict_gap, "_cell_probability", lambda _session, _roi: 0.85)
    base = np.asarray([[853, -1, -1], [999, -1, -1]], dtype=int)

    candidates = strict_gap.strict_gated_gap_edge_candidates(
        base,
        sessions=(object(), object(), object()),
        feature_index={
            (0, 2, 853, 554): _feature(),
            (0, 2, 777, 555): _feature(),
        },
        gate_config=strict_gap.StrictGapGateConfig(),
        seed_rois={853, 999},
    )

    assert candidates == (
        strict_gap.StrictGapCandidate(
            edge=(0, 2, 853, 554),
            candidate_track_id=0,
            accepted=True,
            reason="accepted",
        ),
    )


def test_apply_strict_gated_gap_edges_inserts_target_observation() -> None:
    base = np.asarray([[853, -1, -1]], dtype=int)
    candidates = (
        strict_gap.StrictGapCandidate(
            edge=(0, 2, 853, 554),
            candidate_track_id=0,
            accepted=True,
            reason="accepted",
        ),
    )

    output = strict_gap.apply_strict_gated_gap_edges(base, candidates)

    np.testing.assert_array_equal(output, [[853, -1, 554]])


def test_apply_strict_gated_gap_edges_rejects_duplicate_target() -> None:
    base = np.asarray([[853, -1, -1], [999, -1, 554]], dtype=int)
    candidates = (
        strict_gap.StrictGapCandidate(
            edge=(0, 2, 853, 554),
            candidate_track_id=0,
            accepted=True,
            reason="accepted",
        ),
    )

    output = strict_gap.apply_strict_gated_gap_edges(base, candidates)

    np.testing.assert_array_equal(output, base)


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
