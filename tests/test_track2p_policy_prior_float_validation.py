"""Regression tests for Track2p-policy prior scalar validation."""

from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.track2p_policy_priors import Track2pPolicyPriorConfig


@pytest.mark.parametrize(
    "field_name",
    [
        "relief",
        "accepted_cost_cap",
        "non_policy_penalty",
        "min_cost",
        "rescue_min_iou",
        "rescue_margin",
        "large_cost",
    ],
)
def test_policy_prior_config_rejects_boolean_float_parameters(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        Track2pPolicyPriorConfig(**{field_name: True})


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"relief": "0.25"}, 0.25),
        ({"accepted_cost_cap": np.float64(1.5)}, 1.5),
        ({"rescue_min_iou": "1.0"}, 1.0),
    ],
)
def test_policy_prior_config_keeps_numeric_float_coercion(
    kwargs: dict[str, object], expected: float
) -> None:
    config = Track2pPolicyPriorConfig(**kwargs)
    field_name = next(iter(kwargs))

    assert getattr(config, field_name) == expected
