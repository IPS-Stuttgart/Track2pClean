from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from bayescatrack.association import pyrecest_global_assignment as assignment


@pytest.mark.parametrize("max_gap", [True, 0, 1.5, "2.5", float("nan")])
def test_session_edge_pairs_rejects_invalid_max_gap(max_gap: object) -> None:
    with pytest.raises(ValueError, match="max_gap"):
        assignment.session_edge_pairs(3, max_gap=max_gap)


@pytest.mark.parametrize("num_sessions", [True, -1, 2.5, "3.5", float("inf")])
def test_session_edge_pairs_rejects_invalid_num_sessions(num_sessions: object) -> None:
    with pytest.raises(ValueError, match="num_sessions"):
        assignment.session_edge_pairs(num_sessions, max_gap=1)


def test_session_edge_pairs_accepts_integer_like_inputs() -> None:
    assert assignment.session_edge_pairs("4", max_gap="2") == (
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    )


@dataclass(frozen=True)
class _DummySolverResult:
    tracks: tuple[dict[int, int], ...] = ()


def test_global_assignment_validates_and_normalizes_edge_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
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


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("start_cost", True),
        ("end_cost", np.bool_(False)),
        ("gap_penalty", -0.1),
        ("cost_threshold", -1.0),
        ("start_cost", float("nan")),
        ("end_cost", float("inf")),
        ("gap_penalty", "not-a-number"),
    ],
)
def test_global_assignment_rejects_invalid_solver_controls(
    field: str,
    bad_value: object,
) -> None:
    kwargs: dict[str, object] = {
        "start_cost": 1.0,
        "end_cost": 1.0,
        "gap_penalty": 0.0,
        "cost_threshold": None,
    }
    kwargs[field] = bad_value

    with pytest.raises(ValueError, match=field):
        assignment.solve_global_assignment_from_pairwise_costs(
            {(0, 1): np.zeros((2, 3))},
            session_sizes=(2, 3),
            **kwargs,
        )


def test_global_assignment_accepts_zero_solver_controls(monkeypatch: pytest.MonkeyPatch):
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
        del costs, session_sizes
        calls.update(
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

    assignment.solve_global_assignment_from_pairwise_costs(
        {(0, 1): np.zeros((2, 3))},
        session_sizes=(2, 3),
        start_cost=0,
        end_cost=0.0,
        gap_penalty="0",
        cost_threshold=0,
    )

    assert calls == {
        "start_cost": 0.0,
        "end_cost": 0.0,
        "gap_penalty": 0.0,
        "cost_threshold": 0.0,
    }


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
