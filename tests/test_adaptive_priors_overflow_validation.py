from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from bayescatrack.association.adaptive_priors import (
    AdaptiveEdgePriorConfig,
    fit_gap_costs_from_reference,
)


class _ReferenceForGapCosts:
    def __init__(self) -> None:
        self._matrix = np.asarray([[0, 0]], dtype=object)
        self.n_sessions = 2

    def filtered_indices(self, *, curated_only: bool = False) -> np.ndarray:
        return self._matrix


def _overflowing_fraction() -> Fraction:
    return Fraction(10**10000, 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"session_gap_weight": _overflowing_fraction()},
            "session_gap_weight must be finite and non-negative",
        ),
        (
            {"large_cost": _overflowing_fraction()},
            "large_cost must be a positive finite value",
        ),
        (
            {"learned_gap_costs": {1: _overflowing_fraction()}},
            "learned_gap_costs value must be finite and non-negative",
        ),
        (
            {"learned_gap_costs": {_overflowing_fraction(): 1.0}},
            "learned_gap_costs key must be a positive integer",
        ),
    ],
)
def test_adaptive_prior_config_normalizes_overflowing_numeric_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveEdgePriorConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_gap": _overflowing_fraction()}, "max_gap must be a positive integer"),
        (
            {"smoothing": _overflowing_fraction()},
            "smoothing must be a positive finite value",
        ),
    ],
)
def test_gap_cost_fitting_normalizes_overflowing_numeric_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_gap_costs_from_reference(_ReferenceForGapCosts(), **kwargs)
