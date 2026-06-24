import pytest

from bayescatrack.association.multi_hypothesis import (
    candidate_edge_map,
    consensus_edges,
    edge_union_costs,
    enumerate_track_hypotheses,
    top_k_edge_candidates,
)


def test_top_k_edge_candidates_rejects_silent_session_edge_coercion() -> None:
    with pytest.raises(ValueError, match="edge source session must be a non-negative integer"):
        top_k_edge_candidates([[0.0]], edge=(False, 1))


@pytest.mark.parametrize(
    "matrix_values",
    [
        (((0, 1, 0, 1.5),),),
        (((0, 1, 0, True),),),
    ],
)
def test_consensus_edges_rejects_silent_edge_set_index_coercion(matrix_values) -> None:
    with pytest.raises(ValueError, match="track matrices or edge sets entries must be integers"):
        consensus_edges(matrix_values, min_votes=1)


def test_consensus_edges_rejects_silent_track_matrix_index_coercion() -> None:
    with pytest.raises(ValueError, match="track matrices or edge sets entries must be integers"):
        consensus_edges(([[10, 11.5]],), min_votes=1)


def test_consensus_edges_rejects_negative_non_fill_track_indices() -> None:
    with pytest.raises(ValueError, match="non-negative ROI indices or the fill value"):
        consensus_edges(([[10, -2]],), min_votes=1)


def test_candidate_edge_map_rejects_silent_roi_index_coercion() -> None:
    with pytest.raises(ValueError, match="source ROI indices entry must be a non-negative integer"):
        candidate_edge_map({(0, 1): [[0.0]]}, [[0.5], [2]])


def test_enumerate_track_hypotheses_rejects_silent_start_index_coercion() -> None:
    with pytest.raises(ValueError, match="start ROI index must be a non-negative integer"):
        enumerate_track_hypotheses(("s0", "s1"), {}, start_roi_indices=[True])


def test_edge_union_costs_rejects_silent_edge_key_coercion() -> None:
    with pytest.raises(ValueError, match="edge target ROI must be a non-negative integer"):
        edge_union_costs(({(0, 1, 0, 1.5): 1},))
