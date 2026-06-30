import importlib

import numpy as np


def test_pairwise_margin_components_return_expected_shape():
    advanced = importlib.import_module("bayescatrack.advanced_roi_components")
    components = advanced.pairwise_cost_margin_components(
        np.zeros((2, 2), dtype=float),
    )

    assert components["ambiguity_margin_cost"].shape == (2, 2)
