from __future__ import annotations

import numpy as np
from bayescatrack.advanced_roi_components import candidate_mask_from_cost_matrix


def test_candidate_pruning_accepts_empty_rows_with_margin_gate():
    mask = candidate_mask_from_cost_matrix(
        np.zeros((0, 3), dtype=float),
        top_k=None,
        gate_margin=1.0,
    )

    assert mask.shape == (0, 3)
    assert mask.dtype == bool


def test_candidate_pruning_accepts_empty_columns_with_margin_gate():
    mask = candidate_mask_from_cost_matrix(
        np.zeros((3, 0), dtype=float),
        top_k=1,
        gate_margin=1.0,
    )

    assert mask.shape == (3, 0)
    assert mask.dtype == bool
