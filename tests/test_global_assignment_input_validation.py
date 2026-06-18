from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from bayescatrack.association import pyrecest_global_assignment as assignment


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


@pytest.mark.parametrize(
    "max_gap",
    [True, False, 0, -1, 1.5, "2", float("nan"), float("inf")],
)
def test_session_edge_pairs_rejects_invalid_max_gap(max_gap: object) -> None:
    with pytest.raises(ValueError, match="max_gap"):
        assignment.session_edge_pairs(3, max_gap=max_gap)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "num_sessions",
    [True, False, -1, 1.5, "3", float("nan"), float("inf")],
)
def test_session_edge_pairs_rejects_invalid_num_sessions(num_sessions: object) -> None:
    with pytest.raises(ValueError, match="num_sessions"):
        assignment.session_edge_pairs(num_sessions)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("triplet_weight", True),
        ("triplet_weight", False),
        ("triplet_weight", float("nan")),
        ("triplet_weight", float("inf")),
        ("triplet_weight", -0.1),
        ("support_cost_cap", True),
        ("support_cost_cap", False),
        ("support_cost_cap", float("nan")),
        ("support_cost_cap", float("inf")),
        ("support_cost_cap", -0.1),
        ("max_penalty", True),
        ("max_penalty", False),
        ("max_penalty", float("nan")),
        ("max_penalty", float("inf")),
        ("max_penalty", -0.1),
    ],
)
def test_triplet_support_config_rejects_invalid_float_controls(
    field: str, value: float | bool
) -> None:
    with pytest.raises(ValueError, match=field):
        assignment.TripletSupportConsistencyConfig(**{field: value})


@pytest.mark.parametrize(
    "support_top_k",
    [True, False, 0, -1, 1.5, "2", float("nan"), float("inf")],
)
def test_triplet_support_config_rejects_invalid_support_top_k(
    support_top_k: object,
) -> None:
    with pytest.raises(ValueError, match="support_top_k"):
        assignment.TripletSupportConsistencyConfig(
            support_top_k=support_top_k  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_cost", True),
        ("start_cost", False),
        ("start_cost", float("nan")),
        ("start_cost", float("inf")),
        ("end_cost", True),
        ("end_cost", False),
        ("end_cost", float("nan")),
        ("end_cost", float("inf")),
        ("gap_penalty", True),
        ("gap_penalty", False),
        ("gap_penalty", float("nan")),
        ("gap_penalty", float("inf")),
        ("cost_threshold", True),
        ("cost_threshold", False),
        ("cost_threshold", float("nan")),
        ("cost_threshold", float("inf")),
    ],
)
def test_global_assignment_rejects_invalid_solver_prior_controls(
    field: str, value: float | bool
) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        assignment.solve_global_assignment_from_pairwise_costs(
            {(0, 1): np.zeros((2, 3))},
            session_sizes=(2, 3),
            **kwargs,
        )
