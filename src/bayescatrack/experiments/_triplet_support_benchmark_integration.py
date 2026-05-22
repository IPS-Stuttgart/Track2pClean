"""Activate legacy triplet-support benchmark knobs.

The Track2p benchmark exposes ``triplet_weight``/``support_*`` settings and the
association layer implements the corresponding skip-edge support penalty.  This
small integration keeps the benchmark API stable while making those options take
effect for configured global-assignment runs.
"""

from __future__ import annotations

from typing import Any, Sequence

from bayescatrack.association.pyrecest_global_assignment import (
    apply_triplet_support_consistency,
    solve_global_assignment_from_pairwise_costs,
)
from bayescatrack.core.bridge import Track2pSession


def install_track2p_benchmark_triplet_support_integration() -> None:
    """Install an idempotent wrapper around the benchmark solver helper."""

    from bayescatrack.experiments import track2p_benchmark as benchmark

    original = benchmark.solve_configured_global_assignment
    if getattr(original, "_bayescatrack_triplet_support_integration", False):
        return

    def _solve_configured_global_assignment_with_triplet_support(
        sessions: Sequence[Track2pSession],
        config: Any,
        *,
        cost: Any | None = None,
        calibrated_model: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        assignment = original(
            sessions,
            config,
            cost=cost,
            calibrated_model=calibrated_model,
            **kwargs,
        )
        triplet_config = benchmark._triplet_support_consistency_config(
            config
        )  # pylint: disable=protected-access
        if triplet_config is None:
            return assignment
        adjusted_pairwise_costs = apply_triplet_support_consistency(
            assignment.pairwise_costs,
            config=triplet_config,
        )
        return solve_global_assignment_from_pairwise_costs(
            adjusted_pairwise_costs,
            session_sizes=assignment.session_sizes,
            session_edges=assignment.session_edges,
            start_cost=float(config.start_cost),
            end_cost=float(config.end_cost),
            gap_penalty=float(config.gap_penalty),
            cost_threshold=config.cost_threshold,
        )

    setattr(
        _solve_configured_global_assignment_with_triplet_support,
        "_bayescatrack_triplet_support_integration",
        True,
    )
    setattr(
        _solve_configured_global_assignment_with_triplet_support,
        "_bayescatrack_original",
        original,
    )
    benchmark.solve_configured_global_assignment = (  # type: ignore[assignment]
        _solve_configured_global_assignment_with_triplet_support
    )
