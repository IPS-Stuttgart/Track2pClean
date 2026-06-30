import importlib

import numpy as np


def test_pairwise_margin_components_reject_zero_large_cost():
    advanced = importlib.import_module("bayescatrack.advanced_roi_components")

    try:
        advanced.pairwise_cost_margin_components(
            np.zeros((2, 2), dtype=float),
            large_cost=0.0,
        )
    except ValueError as exc:
        assert "large_cost must be a finite positive value" in str(exc)
    else:
        raise AssertionError("expected invalid large_cost to be rejected")
