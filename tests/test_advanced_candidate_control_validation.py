from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.advanced_roi_components import (
    CandidatePruningConfig,
    candidate_mask_from_cost_matrix,
)


class _OverflowIndex:
    def __index__(self) -> int:
        raise OverflowError("synthetic overflow")


class _ValueIndex:
    def __index__(self) -> int:
        raise ValueError("synthetic value error")


def test_advanced_top_k_controls_accept_text_integer_values_for_compatibility():
    value = str(1)
    mask = candidate_mask_from_cost_matrix(np.zeros((2, 2), dtype=float), top_k=value)

    assert CandidatePruningConfig(top_k_per_roi=value).top_k_per_roi == 1
    assert mask.shape == (2, 2)


def test_advanced_top_k_controls_accept_numeric_integer_scalars():
    mask = candidate_mask_from_cost_matrix(
        np.asarray([[0.0, 10.0], [10.0, 0.0]], dtype=float),
        top_k=1.0,
    )

    assert CandidatePruningConfig(top_k_per_roi=1.0).top_k_per_roi == 1
    assert mask.tolist() == [[True, False], [False, True]]


def test_advanced_top_k_controls_normalize_bad_index_protocol_errors():
    for bad_value in (_OverflowIndex(), _ValueIndex()):
        with pytest.raises(ValueError, match="top_k_per_roi must be a positive integer or None"):
            CandidatePruningConfig(top_k_per_roi=bad_value)
        with pytest.raises(ValueError, match="top_k must be a positive integer or None"):
            candidate_mask_from_cost_matrix(np.zeros((2, 2), dtype=float), top_k=bad_value)
