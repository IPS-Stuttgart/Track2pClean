from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.adaptive_priors import AdaptiveEdgePriorConfig


@pytest.mark.parametrize(
    ("learned_gap_costs", "message"),
    [
        ({0: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({-1: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({1.5: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({np.inf: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({True: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({1: np.nan}, "learned_gap_costs value must be finite and non-negative"),
        ({1: np.inf}, "learned_gap_costs value must be finite and non-negative"),
        ({1: -0.1}, "learned_gap_costs value must be finite and non-negative"),
        ({1: False}, "learned_gap_costs value must be finite and non-negative"),
    ],
)
def test_adaptive_edge_prior_config_rejects_invalid_learned_gap_costs(
    learned_gap_costs: dict[object, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveEdgePriorConfig(learned_gap_costs=learned_gap_costs)


def test_adaptive_edge_prior_config_coerces_valid_learned_gap_costs() -> None:
    config = AdaptiveEdgePriorConfig(learned_gap_costs={"1": "0.25", 2: 0.5})

    assert config.learned_gap_costs == {1: 0.25, 2: 0.5}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"large_cost": np.nan}, "large_cost must be a positive finite value"),
        ({"large_cost": np.inf}, "large_cost must be a positive finite value"),
        ({"large_cost": 0.0}, "large_cost must be a positive finite value"),
        ({"large_cost": True}, "large_cost must be a positive finite value"),
    ],
)
def test_adaptive_edge_prior_config_rejects_invalid_large_cost(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveEdgePriorConfig(**kwargs)
