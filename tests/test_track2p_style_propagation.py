from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.pyrecest_global_assignment import (
    solve_track2p_style_propagation_from_pairwise_costs,
)


def test_track2p_style_propagation_restricts_to_seed_rows_and_ignores_skip_edges():
    pairwise_costs = {
        (0, 1): np.array(
            [
                [0.1, 5.0, 5.0],
                [5.0, 0.2, 5.0],
                [5.0, 5.0, 0.3],
            ]
        ),
        (1, 2): np.array(
            [
                [0.1, 5.0, 5.0],
                [5.0, 5.0, 0.2],
                [5.0, 0.3, 5.0],
            ]
        ),
        (0, 2): np.zeros((3, 3), dtype=float),
    }

    run = solve_track2p_style_propagation_from_pairwise_costs(
        pairwise_costs,
        session_sizes=(3, 3, 3),
        seed_detection_indices=[1],
        cost_threshold=1.0,
    )

    assert run.session_edges == ((0, 1), (1, 2))
    assert run.result.tracks == ({0: 1, 1: 1, 2: 2},)
    assert run.result.total_cost == pytest.approx(0.4)


def test_track2p_style_propagation_stops_when_next_link_is_thresholded():
    run = solve_track2p_style_propagation_from_pairwise_costs(
        {
            (0, 1): np.array([[0.1]]),
            (1, 2): np.array([[2.0]]),
        },
        session_sizes=(1, 1, 1),
        cost_threshold=1.0,
    )

    assert run.result.tracks == ({0: 0, 1: 0},)
    assert run.result.total_cost == pytest.approx(0.1)


def test_track2p_style_propagation_can_grow_backward_from_later_seed_session():
    run = solve_track2p_style_propagation_from_pairwise_costs(
        {
            (0, 1): np.array(
                [
                    [4.0, 0.1],
                    [0.2, 4.0],
                ]
            ),
            (1, 2): np.array(
                [
                    [4.0, 0.3],
                    [0.4, 4.0],
                ]
            ),
        },
        session_sizes=(2, 2, 2),
        seed_session=1,
        seed_detection_indices=[1],
        cost_threshold=1.0,
    )

    assert run.result.tracks == ({1: 1, 2: 0, 0: 0},)
    assert run.result.total_cost == pytest.approx(0.5)
