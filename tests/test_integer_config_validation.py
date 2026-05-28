"""Regression tests for strict integer-like configuration parsing."""

import numpy as np
import pytest
from bayescatrack.association.shifted_overlap import (
    pairwise_shifted_overlap_matrices,
    shift_offsets,
)
from bayescatrack.association.track2p_policy_priors import Track2pPolicyPriorConfig


@pytest.mark.parametrize("bad_radius", [True, False, 1.5, "1.5", np.nan])
def test_shifted_overlap_radius_rejects_bool_fractional_and_nonfinite_values(
    bad_radius,
):
    reference = np.zeros((1, 4, 4), dtype=bool)
    measurement = np.zeros((1, 4, 4), dtype=bool)
    reference[0, 1, 1] = True
    measurement[0, 1, 2] = True

    with pytest.raises(ValueError, match="radius must be an integer"):
        pairwise_shifted_overlap_matrices(
            reference,
            measurement,
            radius=bad_radius,
        )


@pytest.mark.parametrize("integer_like_radius", [1, np.int64(1), 1.0, "1"])
def test_shifted_overlap_radius_accepts_integer_like_values(integer_like_radius):
    offsets = shift_offsets(integer_like_radius)

    assert offsets[0] == (0, 0)
    assert (0, 1) in offsets


@pytest.mark.parametrize("bad_top_k", [True, False, 1.5, "1.5", np.nan])
def test_track2p_policy_top_k_rejects_bool_fractional_and_nonfinite_values(
    bad_top_k,
):
    with pytest.raises(ValueError, match="row_top_k"):
        Track2pPolicyPriorConfig(row_top_k=bad_top_k)
    with pytest.raises(ValueError, match="column_top_k"):
        Track2pPolicyPriorConfig(column_top_k=bad_top_k)


@pytest.mark.parametrize("bad_max_gap", [True, False, 0, -1, 1.5, "1.5", np.nan])
def test_track2p_policy_max_gap_rejects_invalid_integer_values(bad_max_gap):
    with pytest.raises(ValueError, match="max_gap"):
        Track2pPolicyPriorConfig(max_gap=bad_max_gap)


@pytest.mark.parametrize("integer_like_value", [1, np.int64(1), 1.0, "1"])
def test_track2p_policy_accepts_integer_like_top_k_and_gap_values(
    integer_like_value,
):
    config = Track2pPolicyPriorConfig(
        row_top_k=integer_like_value,
        column_top_k=integer_like_value,
        max_gap=integer_like_value,
    )

    assert config.row_top_k == 1
    assert config.column_top_k == 1
    assert config.max_gap == 1
