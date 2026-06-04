from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_growth_veto_whatif as veto


def test_growth_veto_whatif_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-growth-veto-whatif"]

    assert canonical == "track2p-policy-growth-veto-whatif"
    assert cli._BENCHMARK_ALIASES["track2p-component-growth-veto-whatif"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == "bayescatrack.experiments.track2p_policy_growth_veto_whatif"


def test_growth_veto_parser_exposes_defaults() -> None:
    args = veto.build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "growth_veto_edges.csv",
        ]
    )

    assert args.threshold_method == "min"
    assert args.iou_distance_threshold == 12.0
    assert args.anchor_min_registered_iou == 0.50
    assert args.anchor_min_shifted_iou == 0.30
    assert args.anchor_min_cell_probability == 0.80
    assert args.progress is False


def test_remove_edge_occurrence_splits_component_without_gt_guard() -> None:
    predicted = np.asarray([[10, 11, 12, 13]], dtype=int)

    result = veto._remove_edge_occurrence(predicted, (1, 2, 11, 12), occurrence_index=0)

    assert result.would_split_component == 1
    assert result.complete_component_size == 4
    assert result.is_terminal_edge == 0
    assert result.is_last_session_edge == 0
    assert result.tracks.tolist() == [[10, 11, -1, -1], [-1, -1, 12, 13]]


def test_remove_edge_occurrence_marks_terminal_last_session_edge() -> None:
    predicted = np.asarray([[10, 11, 12, 13]], dtype=int)

    result = veto._remove_edge_occurrence(predicted, (2, 3, 12, 13), occurrence_index=0)

    assert result.is_terminal_edge == 1
    assert result.is_last_session_edge == 1
    assert result.tracks.tolist() == [[10, 11, 12, -1], [-1, -1, -1, 13]]


def test_edge_source_classifies_incremental_support() -> None:
    edge = (0, 1, 10, 11)

    assert (
        veto._edge_source(
            edge,
            occurrence_index=0,
            policy_counts=Counter({edge: 1}),
            cleanup_counts=Counter({edge: 1}),
            suffix_counts=Counter({edge: 1}),
        )
        == "policy"
    )
    assert (
        veto._edge_source(
            edge,
            occurrence_index=0,
            policy_counts=Counter(),
            cleanup_counts=Counter({edge: 1}),
            suffix_counts=Counter({edge: 1}),
        )
        == "component"
    )
    assert (
        veto._edge_source(
            edge,
            occurrence_index=0,
            policy_counts=Counter(),
            cleanup_counts=Counter(),
            suffix_counts=Counter({edge: 1}),
        )
        == "suffix"
    )
    assert (
        veto._edge_source(
            edge,
            occurrence_index=0,
            policy_counts=Counter(),
            cleanup_counts=Counter(),
            suffix_counts=Counter(),
        )
        == "teacher"
    )


def test_anchor_edges_use_policy_diagnostics_without_feature_cache(monkeypatch) -> None:
    sessions = (object(), object())
    diagnostic = SimpleNamespace(
        session_index=0,
        local_roi_a=0,
        local_roi_b=0,
        assigned_iou=0.75,
    )

    def fake_roi_indices(session: object) -> np.ndarray:
        return np.asarray([10]) if session is sessions[0] else np.asarray([11])

    monkeypatch.setattr(veto, "_roi_indices", fake_roi_indices)
    monkeypatch.setattr(veto, "_cell_probability", lambda *args: 0.90)

    anchors = veto._anchor_edges_from_policy_diagnostics(
        sessions,
        diagnostics=(diagnostic,),
        track2p=np.asarray([[10, 11]], dtype=int),
        component_cleanup=np.asarray([[10, 11]], dtype=int),
        combined=np.asarray([[10, 11]], dtype=int),
        min_registered_iou=0.50,
        min_cell_probability=0.80,
    )

    assert anchors == {(0, 1): ((0, 1, 10, 11),)}


def test_policy_feature_index_from_diagnostics(monkeypatch) -> None:
    sessions = (object(), object())
    diagnostic = SimpleNamespace(
        session_index=0,
        local_roi_a=0,
        local_roi_b=0,
        assigned_iou=0.72,
        centroid_distance=3.5,
        area_ratio=0.91,
        row_margin=0.12,
        column_margin=0.08,
        threshold=0.40,
        threshold_margin=0.32,
    )

    def fake_roi_indices(session: object) -> np.ndarray:
        return np.asarray([10]) if session is sessions[0] else np.asarray([11])

    monkeypatch.setattr(veto, "_roi_indices", fake_roi_indices)
    monkeypatch.setattr(veto, "_cell_probability", lambda *args: 0.90)

    index = veto._policy_feature_index_from_diagnostics(
        sessions,
        diagnostics=(diagnostic,),
    )

    assert set(index) == {(0, 1, 10, 11)}
    feature = index[(0, 1, 10, 11)]
    assert feature.registered_iou == 0.72
    assert feature.shifted_iou != feature.shifted_iou
    assert feature.row_rank == 1
    assert feature.column_rank == 1
    assert feature.threshold_margin == 0.32


def test_sparse_edge_feature_augmentation_keeps_existing_suffix_shifted_iou() -> None:
    rows = [
        {"session_a": 0, "session_b": 1, "roi_a": 10, "roi_b": 11},
        {"session_a": 1, "session_b": 2, "roi_a": 11, "roi_b": 12},
    ]
    features = {
        (0, 1, 10, 11): veto._AcceptedEdgeFeature(
            registered_iou=0.4,
            shifted_iou=0.5,
            centroid_distance=2.0,
            area_ratio=0.9,
            cell_probability_a=0.8,
            cell_probability_b=0.85,
            row_rank=2,
            column_rank=3,
            row_margin=0.1,
            column_margin=0.2,
            threshold=0.3,
            threshold_margin=0.1,
            assigned_by_hungarian=0,
        )
    }

    augmented = veto._augment_with_sparse_edge_features(rows, (), features)

    assert augmented[0]["shifted_iou"] == 0.5
    assert augmented[0]["registered_iou"] == 0.4
    assert augmented[0]["cell_probability_a"] == 0.8
    assert augmented[0]["cell_probability_b"] == 0.85
    assert augmented[1]["shifted_iou"] != augmented[1]["shifted_iou"]
    assert augmented[1]["registered_iou"] != augmented[1]["registered_iou"]
