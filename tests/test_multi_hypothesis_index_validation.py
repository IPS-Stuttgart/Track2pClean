import pytest

from bayescatrack.association.multi_hypothesis import (
    candidate_edge_map,
    consensus_edges,
    edge_union_costs,
    enumerate_track_hypotheses,
    top_k_edge_candidates,
)


def test_top_k_edge_candidates_rejects_silent_session_edge_coercion() -> None:
    with pytest.raises(ValueError, match=r"edge source_session must"):
        top_k_edge_candidates([[0.0]], edge=(False, 1))


@pytest.mark.parametrize(
    "matrix_values",
    [
        (((0, 1, 0, 1.5),),),
        (((0, 1, 0, True),),),
    ],
)
def test_consensus_edges_rejects_silent_edge_set_index_coercion(matrix_values) -> None:
    with pytest.raises(ValueError, match=r"track matrices or edge sets.*integer"):
        consensus_edges(matrix_values, min_votes=1)


def test_consensus_edges_rejects_silent_track_matrix_index_coercion() -> None:
    with pytest.raises(ValueError, match=r"track matrices or edge sets.*integer"):
        consensus_edges(([[10, 11.5]],), min_votes=1)


def test_consensus_edges_rejects_negative_non_fill_track_indices() -> None:
    with pytest.raises(ValueError, match=r"non-negative ROI indices or fill_value"):
        consensus_edges(([[10, -2]],), min_votes=1)


def test_candidate_edge_map_rejects_silent_roi_index_coercion() -> None:
    with pytest.raises(ValueError, match=r"roi_indices_by_session.*integer"):
        candidate_edge_map({(0, 1): [[0.0]]}, [[0.5], [2]])


def test_enumerate_track_hypotheses_rejects_silent_start_index_coercion() -> None:
    with pytest.raises(ValueError, match=r"start_roi_indices.*must"):
        enumerate_track_hypotheses(("s0", "s1"), {}, start_roi_indices=[True])


def test_edge_union_costs_rejects_silent_edge_key_coercion() -> None:
    with pytest.raises(ValueError, match=r"edge target_roi must be a non-negative integer"):
        edge_union_costs(({(0, 1, 0, 1.5): 1},))
