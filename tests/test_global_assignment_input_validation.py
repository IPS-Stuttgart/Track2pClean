from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from bayescatrack.association import pyrecest_global_assignment as assignment


@dataclass(frozen=True)
class _DummySolverResult:
    tracks: tuple[dict[int, int], ...] = ()


def test_global_assignment_validates_and_normalizes_edge_metadata(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, Any] = {}

    def fake_solver(
        costs: dict[tuple[int, int], np.ndarray],
        *,
        session_sizes: tuple[int, ...],
        start_cost: float,
        end_cost: float,
        gap_penalty: float,
        cost_threshold: float | None,
    ) -> _DummySolverResult:
        calls.update(
            costs=costs,
            session_sizes=session_sizes,
            start_cost=start_cost,
            end_cost=end_cost,
            gap_penalty=gap_penalty,
            cost_threshold=cost_threshold,
        )
        return _DummySolverResult()

    monkeypatch.setattr(
        assignment,
        "_load_pyrecest_multisession_solver",
        lambda: fake_solver,
    )

    run = assignment.solve_global_assignment_from_pairwise_costs(
        {
            (0.0, "1"): np.zeros((2, 3)),
            (1, 2): np.ones((3, 1)),
        },
        session_sizes=[2.0, "3", 1],
        session_edges=[(0, 1), (1.0, "2")],
        start_cost=2.5,
        end_cost=3.5,
        gap_penalty=0.25,
        cost_threshold=None,
    )

    assert run.session_sizes == (2, 3, 1)
    assert run.session_edges == ((0, 1), (1, 2))
    assert tuple(run.pairwise_costs) == ((0, 1), (1, 2))
    assert calls["session_sizes"] == (2, 3, 1)
    assert tuple(calls["costs"]) == ((0, 1), (1, 2))
    assert calls["start_cost"] == 2.5
    assert calls["end_cost"] == 3.5
    assert calls["gap_penalty"] == 0.25
    assert calls["cost_threshold"] is None


def test_global_assignment_rejects_stale_session_edges():
    with pytest.raises(ValueError, match="same edges"):
        assignment.solve_global_assignment_from_pairwise_costs(
            {(0, 1): np.zeros((2, 3))},
            session_sizes=(2, 3, 1),
            session_edges=((0, 1), (1, 2)),
        )


def test_global_assignment_rejects_unlisted_pairwise_cost_edges():
    with pytest.raises(ValueError, match="unlisted pairwise costs"):
        assignment.solve_global_assignment_from_pairwise_costs(
            {
                (0, 1): np.zeros((2, 3)),
                (1, 2): np.zeros((3, 1)),
            },
            session_sizes=(2, 3, 1),
            session_edges=((0, 1),),
        )


def test_global_assignment_rejects_cost_matrices_with_wrong_shape():
    with pytest.raises(ValueError, match="must have shape"):
        assignment.solve_global_assignment_from_pairwise_costs(
            {(0, 1): np.zeros((2, 2))},
            session_sizes=(2, 3),
        )


def test_global_assignment_rejects_non_forward_edges():
    with pytest.raises(ValueError, match="point forward"):
        assignment.solve_global_assignment_from_pairwise_costs(
            {(1, 1): np.zeros((3, 3))},
            session_sizes=(2, 3),
        )
