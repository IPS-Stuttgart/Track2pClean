from __future__ import annotations

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
