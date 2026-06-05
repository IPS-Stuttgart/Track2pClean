import numpy as np

from bayescatrack import cli
from bayescatrack.experiments.track2p_policy_growth_regularized_assignment import (
    growth_regularized_cost_matrix,
)


def test_growth_regularized_assignment_cli_aliases() -> None:
    canonical = "track2p-policy-growth-regularized-assignment"
    assert canonical in cli._BENCHMARK_COMMANDS
    assert cli._BENCHMARK_ALIASES["track2p-growth-regularized-assignment"] == canonical
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-growth-regularized-assignment"]
        == canonical
    )


def test_growth_regularized_cost_penalizes_growth_outliers() -> None:
    iou = np.asarray([[0.8, 0.8]], dtype=float)
    mahalanobis = np.asarray([[0.0, 30.0]], dtype=float)
    area_residual = np.asarray([[0.0, 0.5]], dtype=float)

    cost = growth_regularized_cost_matrix(
        iou,
        mahalanobis,
        area_residual,
        lambda_growth=0.10,
        lambda_area=0.05,
        growth_mahalanobis_cap=30.0,
    )

    assert cost[0, 0] < cost[0, 1]
    assert np.isclose(cost[0, 0], 0.2)
    assert np.isclose(cost[0, 1], 0.325)
