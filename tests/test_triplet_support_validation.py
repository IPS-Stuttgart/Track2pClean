from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.pyrecest_global_assignment import (
    TripletSupportConsistencyConfig,
    apply_triplet_support_consistency,
)


def _triplet_costs() -> dict[tuple[int, int], np.ndarray]:
    return {
        (0, 1): np.asarray([[5.0]], dtype=float),
        (1, 2): np.asarray([[5.0]], dtype=float),
        (0, 2): np.asarray([[1.0]], dtype=float),
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (TripletSupportConsistencyConfig(triplet_weight=True), "triplet_weight"),
        (
            TripletSupportConsistencyConfig(
                triplet_weight=1.0,
                support_top_k=True,
            ),
            "support_top_k",
        ),
        (
            TripletSupportConsistencyConfig(
                triplet_weight=1.0,
                support_top_k=1.5,
            ),
            "support_top_k",
        ),
        (
            TripletSupportConsistencyConfig(
                triplet_weight=1.0,
                support_cost_cap=False,
            ),
            "support_cost_cap",
        ),
        (
            TripletSupportConsistencyConfig(
                triplet_weight=1.0,
                max_penalty=True,
            ),
            "max_penalty",
        ),
    ],
)
def test_triplet_support_config_rejects_ambiguous_values(
    config: TripletSupportConsistencyConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_triplet_support_consistency(_triplet_costs(), config=config)


def test_triplet_support_config_normalizes_integer_like_controls() -> None:
    adjusted = apply_triplet_support_consistency(
        _triplet_costs(),
        config=TripletSupportConsistencyConfig(
            triplet_weight="2.0",
            support_top_k="1",
            support_cost_cap="1.0",
            max_penalty="1.5",
        ),
    )

    np.testing.assert_allclose(adjusted[(0, 2)], np.asarray([[2.5]], dtype=float))
