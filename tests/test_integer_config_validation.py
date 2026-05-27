"""Regression tests for strict integer-like configuration parsing."""

import numpy as np
import pytest
from bayescatrack.association.shifted_overlap import (
    pairwise_shifted_overlap_matrices,
    shift_offsets,
)


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
