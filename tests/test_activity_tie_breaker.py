from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.activity_tie_breaker import (
    activity_tie_breaker_cost_matrix,
)


def test_activity_tie_breaker_masks_unavailable_activity_pairs() -> None:
    components = {
        "activity_tiebreaker_cost": np.array(
            [
                [0.0, 1.0],
                [0.25, 0.5],
            ],
            dtype=float,
        ),
        "activity_tiebreaker_available": np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
    }

    adjusted = activity_tie_breaker_cost_matrix(components, weight=0.4)

    npt.assert_allclose(
        adjusted,
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
            ],
            dtype=float,
        ),
    )


@pytest.mark.parametrize(
    "bad_weight",
    [
        True,
        -0.1,
        np.nan,
        np.inf,
        "0.1",
        [0.1],
    ],
)
def test_activity_tie_breaker_rejects_malformed_weight(bad_weight) -> None:
    components = {"activity_tiebreaker_cost": np.zeros((2, 2), dtype=float)}

    with pytest.raises(ValueError, match="weight"):
        activity_tie_breaker_cost_matrix(components, weight=bad_weight)


def test_activity_tie_breaker_keeps_legacy_behavior_without_availability() -> None:
    components = {"custom_cost": np.array([[np.nan, np.inf, -np.inf]], dtype=float)}

    adjusted = activity_tie_breaker_cost_matrix(
        components,
        component_name="custom_cost",
        weight=2.0,
    )

    npt.assert_allclose(adjusted, np.array([[1.0, 2.0e6, 0.0]], dtype=float))


def test_activity_tie_breaker_validates_availability_shape() -> None:
    components = {
        "activity_tiebreaker_cost": np.zeros((2, 2), dtype=float),
        "activity_tiebreaker_available": np.ones((2, 1), dtype=float),
    }

    with pytest.raises(ValueError, match="Availability component"):
        activity_tie_breaker_cost_matrix(components, weight=0.1)
