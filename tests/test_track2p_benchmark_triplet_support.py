from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt

from bayescatrack.association.pyrecest_global_assignment import GlobalAssignmentRun
from bayescatrack.experiments import _triplet_support_benchmark_integration as integration
from bayescatrack.experiments import track2p_benchmark as benchmark_module
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    solve_configured_global_assignment,
)


def test_legacy_triplet_support_knobs_adjust_skip_edges_before_final_solve(monkeypatch):
    base_costs = {
        (0, 1): np.asarray([[0.0, 10.0], [10.0, 0.0]], dtype=float),
        (1, 2): np.asarray([[0.0, 10.0], [10.0, 0.0]], dtype=float),
        (0, 2): np.zeros((2, 2), dtype=float),
    }
    base_assignment = GlobalAssignmentRun(
        result=SimpleNamespace(tracks=[]),
        pairwise_costs=base_costs,
        session_sizes=(2, 2, 2),
        session_edges=((0, 1), (0, 2), (1, 2)),
    )
    seen: dict[str, object] = {}

    def fake_initial_solve(sessions, **kwargs):
        seen["initial_sessions"] = sessions
        seen["initial_kwargs"] = kwargs
        return base_assignment

    def fake_final_solve(pairwise_costs, **kwargs):
        seen["final_pairwise_costs"] = pairwise_costs
        seen["final_kwargs"] = kwargs
        return GlobalAssignmentRun(
            result=SimpleNamespace(tracks=[]),
            pairwise_costs=pairwise_costs,
            session_sizes=tuple(kwargs["session_sizes"]),
            session_edges=tuple(kwargs["session_edges"]),
        )

    monkeypatch.setattr(
        benchmark_module,
        "solve_global_assignment_for_sessions",
        fake_initial_solve,
    )
    monkeypatch.setattr(integration, "solve_global_assignment_from_pairwise_costs", fake_final_solve)
    config = Track2pBenchmarkConfig(
        data=Path("."),
        method="global-assignment",
        triplet_weight=3.0,
        support_top_k=1,
        support_cost_cap=1.0,
        triplet_max_penalty=None,
        start_cost=4.0,
        end_cost=5.0,
        gap_penalty=2.0,
        cost_threshold=7.0,
    )
    sessions = [object(), object(), object()]

    assignment = solve_configured_global_assignment(sessions, config)

    assert assignment is not base_assignment
    assert seen["initial_sessions"] == sessions
    adjusted = seen["final_pairwise_costs"]
    assert isinstance(adjusted, dict)
    npt.assert_allclose(adjusted[(0, 1)], base_costs[(0, 1)])
    npt.assert_allclose(adjusted[(1, 2)], base_costs[(1, 2)])
    npt.assert_allclose(
        adjusted[(0, 2)],
        np.asarray([[0.0, 3.0], [3.0, 0.0]], dtype=float),
    )
    assert seen["final_kwargs"] == {
        "session_sizes": (2, 2, 2),
        "session_edges": ((0, 1), (0, 2), (1, 2)),
        "start_cost": 4.0,
        "end_cost": 5.0,
        "gap_penalty": 2.0,
        "cost_threshold": 7.0,
    }


def test_triplet_support_integration_is_noop_when_weight_is_zero(monkeypatch):
    base_assignment = GlobalAssignmentRun(
        result=SimpleNamespace(tracks=[]),
        pairwise_costs={(0, 1): np.zeros((1, 1), dtype=float)},
        session_sizes=(1, 1),
        session_edges=((0, 1),),
    )

    def fake_initial_solve(_sessions, **_kwargs):
        return base_assignment

    def fail_final_solve(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("triplet-weight zero must not rerun the final solver")

    monkeypatch.setattr(
        benchmark_module,
        "solve_global_assignment_for_sessions",
        fake_initial_solve,
    )
    monkeypatch.setattr(integration, "solve_global_assignment_from_pairwise_costs", fail_final_solve)

    assignment = solve_configured_global_assignment(
        [object(), object()],
        Track2pBenchmarkConfig(data=Path("."), method="global-assignment"),
    )

    assert assignment is base_assignment
