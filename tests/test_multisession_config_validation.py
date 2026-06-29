from decimal import Decimal

import numpy as np
import pytest
from bayescatrack.multisession_tracking import MultisessionTrackingConfig


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("max_session_gap", True),
        ("max_session_gap", 1.5),
        ("weighted_centroids", 1),
        ("velocity_variance", np.nan),
        ("regularization", np.inf),
        ("start_cost", np.nan),
        ("end_cost", -0.1),
        ("gap_penalty", "not-a-number"),
        ("cost_threshold", np.inf),
        ("return_pairwise_components", 1),
    ],
)
def test_multisession_config_rejects_ambiguous_scalar_controls(field_name, bad_value):
    with pytest.raises(ValueError, match=field_name):
        MultisessionTrackingConfig(**{field_name: bad_value})


def test_multisession_config_normalizes_numpy_scalar_controls():
    config = MultisessionTrackingConfig(
        max_session_gap=np.int64(2),
        weighted_centroids=np.bool_(True),
        velocity_variance=np.float64(3.0),
        regularization=np.float64(0.25),
        start_cost=np.float64(0.1),
        end_cost=np.float64(0.2),
        gap_penalty=np.float64(0.3),
        cost_threshold=np.float64(1.5),
        return_pairwise_components=np.bool_(False),
    )

    assert config.max_session_gap == 2
    assert config.weighted_centroids is True
    assert config.velocity_variance == 3.0
    assert config.regularization == 0.25
    assert config.start_cost == 0.1
    assert config.end_cost == 0.2
    assert config.gap_penalty == 0.3
    assert config.cost_threshold == 1.5
    assert config.return_pairwise_components is False


@pytest.mark.parametrize(
    "field_name",
    [
        "velocity_variance",
        "regularization",
        "start_cost",
        "end_cost",
        "gap_penalty",
        "cost_threshold",
    ],
)
def test_multisession_config_normalizes_decimal_overflow_controls(field_name):
    with pytest.raises(ValueError, match=rf"{field_name} must be a finite non-negative value"):
        MultisessionTrackingConfig(**{field_name: Decimal("1e999999999999999999999")})
