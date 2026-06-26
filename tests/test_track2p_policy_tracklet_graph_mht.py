from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("pyrecest")

from bayescatrack.experiments import (  # noqa: E402
    track2p_policy_tracklet_graph_mht as graph_mht,
)


def test_mutual_local_links_require_two_sided_ambiguity_margin():
    scores = np.asarray(
        [
            [2.0, 1.95, 0.1],
            [0.2, 1.0, 0.9],
            [0.1, 0.4, 1.8],
        ],
        dtype=float,
    )
    valid = np.ones_like(scores, dtype=bool)

    links = graph_mht._mutual_local_links(scores, valid, min_margin=0.10)

    assert links == ((2, 2),)


def test_path_enumeration_keeps_explicit_no_join_hypothesis():
    edges = {
        1: [
            graph_mht.TrackletEdge(
                source_id=1,
                target_id=2,
                score=0.9,
                raw_score=1.2,
                gap=1,
                registered_iou=0.5,
                shifted_iou=0.5,
                centroid_distance=2.0,
                area_ratio=0.9,
                growth_residual=1.0,
                growth_mahalanobis=1.0,
                endpoint_cell_probability_min=0.8,
                duplicate_source_rank=1,
                duplicate_target_rank=1,
                would_complete_track=False,
                suspicious_complete_component=False,
                component_incoherence=0.0,
            )
        ]
    }

    paths = graph_mht._enumerate_paths(
        1,
        outgoing=edges,
        graph_config=graph_mht.TrackletGraphConfig(path_hypotheses=4),
    )

    assert graph_mht._PathHypothesis((1,), tuple(), 0.0) in paths
    assert graph_mht._PathHypothesis((1, 2), ((1, 2),), 0.9) in paths


def test_seed_path_selection_enforces_tracklet_conflicts():
    tracklets = (
        graph_mht.Tracklet(0, (10,), 0),
        graph_mht.Tracklet(1, (11,), 0),
        graph_mht.Tracklet(2, (20,), 1),
    )
    edges = (
        graph_mht.TrackletEdge(
            source_id=0,
            target_id=2,
            score=2.0,
            raw_score=2.0,
            gap=1,
            registered_iou=0.7,
            shifted_iou=0.7,
            centroid_distance=1.0,
            area_ratio=1.0,
            growth_residual=0.5,
            growth_mahalanobis=0.5,
            endpoint_cell_probability_min=0.9,
            duplicate_source_rank=1,
            duplicate_target_rank=1,
            would_complete_track=False,
            suspicious_complete_component=False,
            component_incoherence=0.0,
        ),
        graph_mht.TrackletEdge(
            source_id=1,
            target_id=2,
            score=1.5,
            raw_score=1.5,
            gap=1,
            registered_iou=0.7,
            shifted_iou=0.7,
            centroid_distance=1.0,
            area_ratio=1.0,
            growth_residual=0.5,
            growth_mahalanobis=0.5,
            endpoint_cell_probability_min=0.9,
            duplicate_source_rank=1,
            duplicate_target_rank=2,
            would_complete_track=False,
            suspicious_complete_component=False,
            component_incoherence=0.0,
        ),
    )

    paths = graph_mht._select_seed_paths(
        tracklets,
        edges,
        seed_rois=(10, 11),
        seed_session=0,
        graph_config=graph_mht.TrackletGraphConfig(beam_width=8, path_hypotheses=4),
    )

    assert graph_mht._PathHypothesis((0, 2), ((0, 2),), 2.0) in paths
    assert graph_mht._PathHypothesis((1,), tuple(), 0.0) in paths


def test_join_valid_mask_admits_strong_score_frontier_candidate():
    matrices = SimpleNamespace(
        registered_iou=np.asarray([[0.35, 0.35]], dtype=float),
        shifted_iou=np.asarray([[0.60, 0.30]], dtype=float),
        area_ratio=np.asarray([[0.80, 0.80]], dtype=float),
        centroid_distance=np.asarray([[3.0, 3.0]], dtype=float),
        growth_residual=np.asarray([[3.0, 3.0]], dtype=float),
    )
    scores = np.asarray([[0.50, 0.50]], dtype=float)

    valid = graph_mht._join_valid_mask(
        matrices,
        scores,
        graph_mht.TrackletGraphConfig(join_min_edge_score=0.75),
    )

    assert valid.tolist() == [[True, False]]


def test_coverage_audit_splits_candidate_presence_from_solver_rejection():
    reference_tracks = np.asarray([[10, 11, 12, 13]], dtype=int)
    tracklets = (
        graph_mht.Tracklet(0, (10, 11), 0),
        graph_mht.Tracklet(1, (12,), 2),
        graph_mht.Tracklet(2, (13,), 3),
    )
    edges = (
        _edge(0, 1),
        _edge(1, 2),
    )
    selected_paths = (
        graph_mht._PathHypothesis((0, 1, 2), ((1, 2),), 1.0),
    )

    rows, summary = graph_mht._coverage_audit_rows(
        "subject-a",
        reference_tracks,
        tracklets=tracklets,
        edges=edges,
        selected_paths=selected_paths,
        seed_session=0,
    )

    assert summary["tracklet_graph_audit_reference_tracks_with_seed_tracklet"] == 1
    assert summary["tracklet_graph_audit_reference_links"] == 3
    assert summary["tracklet_graph_audit_reference_links_preserved_in_tracklets"] == 1
    assert summary["tracklet_graph_audit_reference_breaks"] == 2
    assert summary["tracklet_graph_audit_break_correct_join_present"] == 2
    assert summary["tracklet_graph_audit_break_correct_join_selected"] == 1
    assert summary["tracklet_graph_audit_break_solver_rejected"] == 1
    assert summary["tracklet_graph_audit_failure_solver_too_conservative"] == 1
    assert summary["tracklet_graph_audit_failure_conflict_issue"] == 0
    assert summary["tracklet_graph_audit_recovered_correct_joins"] == 1
    assert summary["tracklet_graph_audit_tracks_split_3"] == 1
    break_reasons = [
        row["break_reason"] for row in rows if row.get("row_type") == "reference_break"
    ]
    assert break_reasons == [
        "correct_join_present_solver_rejected",
        "correct_join_selected",
    ]
    failure_classes = [
        row["failure_class"]
        for row in rows
        if row.get("row_type") == "reference_break"
    ]
    assert failure_classes == [
        "solver_too_conservative",
        "recovered_correct_join",
    ]


def _edge(source_id: int, target_id: int) -> graph_mht.TrackletEdge:
    return graph_mht.TrackletEdge(
        source_id=source_id,
        target_id=target_id,
        score=1.0,
        raw_score=1.0,
        gap=1,
        registered_iou=0.5,
        shifted_iou=0.5,
        centroid_distance=1.0,
        area_ratio=1.0,
        growth_residual=0.5,
        growth_mahalanobis=0.5,
        endpoint_cell_probability_min=0.9,
        duplicate_source_rank=1,
        duplicate_target_rank=1,
        would_complete_track=False,
        suspicious_complete_component=False,
        component_incoherence=0.0,
    )
