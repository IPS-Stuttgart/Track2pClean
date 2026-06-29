from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association import pyrecest_global_assignment as assignment


class _BadIndex:
    def __index__(self) -> int:
        raise OverflowError("bad integer conversion")


class _BadFloat:
    def __float__(self) -> float:
        raise OverflowError("bad float conversion")


def test_global_assignment_wraps_bad_pairwise_edge_index() -> None:
    with pytest.raises(ValueError, match="pairwise_costs source session index"):
        assignment.solve_global_assignment_from_pairwise_costs(
            {(_BadIndex(), 1): np.zeros((1, 1))},
            session_sizes=(1, 1),
        )


def test_global_assignment_wraps_bad_session_size() -> None:
    with pytest.raises(ValueError, match="session_sizes"):
        assignment.solve_global_assignment_from_pairwise_costs(
            {(0, 1): np.zeros((1, 1))},
            session_sizes=(1, _BadIndex()),
        )


def test_global_assignment_wraps_bad_session_edge_index() -> None:
    with pytest.raises(ValueError, match="session_edges target session index"):
        assignment.solve_global_assignment_from_pairwise_costs(
            {(0, 1): np.zeros((1, 1))},
            session_sizes=(1, 1),
            session_edges=((0, _BadIndex()),),
        )


def test_global_assignment_wraps_bad_solver_scalar() -> None:
    with pytest.raises(ValueError, match="start_cost"):
        assignment.solve_global_assignment_from_pairwise_costs(
            {(0, 1): np.zeros((1, 1))},
            session_sizes=(1, 1),
            start_cost=_BadFloat(),
        )
