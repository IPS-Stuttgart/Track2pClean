from __future__ import annotations

import numpy as np

from bayescatrack.advanced_roi_components import CandidatePruningConfig, candidate_mask_from_cost_matrix


def test_advanced_top_k_controls_reject_text_values():
    value = str(1)
    try:
        CandidatePruningConfig(top_k_per_roi=value)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    try:
        candidate_mask_from_cost_matrix(np.zeros((2, 2), dtype=float), top_k=value)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_advanced_top_k_controls_accept_numeric_integer_scalars():
    mask = candidate_mask_from_cost_matrix(
        np.asarray([[0.0, 10.0], [10.0, 0.0]], dtype=float),
        top_k=1.0,
    )

    assert CandidatePruningConfig(top_k_per_roi=1.0).top_k_per_roi == 1
    assert mask.tolist() == [[True, False], [False, True]]
