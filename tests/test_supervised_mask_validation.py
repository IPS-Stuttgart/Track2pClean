from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.calibrated_costs import (
    ReferencePairwiseExamples,
    _validated_supervised_mask,
)


def _example_block(supervised_mask: object) -> ReferencePairwiseExamples:
    return ReferencePairwiseExamples(
        session_a=0,
        session_b=1,
        features=np.zeros((2, 2, 1), dtype=float),
        labels=np.eye(2, dtype=int),
        reference_roi_indices=np.arange(2),
        measurement_roi_indices=np.arange(2),
        feature_names=("centroid_distance",),
        supervised_mask=supervised_mask,
    )


def test_validated_supervised_mask_accepts_boolean_values() -> None:
    mask = np.array([[True, False], [np.bool_(False), np.bool_(True)]], dtype=object)

    validated = _validated_supervised_mask(_example_block(mask), (2, 2))

    assert validated.dtype == np.dtype(bool)
    np.testing.assert_array_equal(validated, [[True, False], [False, True]])


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[1, 0], [0, 1]], dtype=int),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([["yes", "no"], ["no", "yes"]], dtype=object),
    ],
)
def test_validated_supervised_mask_rejects_silent_boolean_coercions(mask: object) -> None:
    with pytest.raises(ValueError, match="supervised_mask must contain only boolean values"):
        _validated_supervised_mask(_example_block(mask), (2, 2))
