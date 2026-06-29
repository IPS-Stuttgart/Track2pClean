from __future__ import annotations

import numpy as np
import pytest

from bayescatrack._tracking_start_roi_availability_validation import (
    _normalize_roi_index_sequence,
)


def test_start_roi_availability_rejects_nested_roi_index_sequences():
    with pytest.raises(ValueError, match="one-dimensional sequence"):
        _normalize_roi_index_sequence(
            [[0, 1]],
            field_name="start_roi_indices",
        )

    with pytest.raises(ValueError, match="one-dimensional sequence"):
        _normalize_roi_index_sequence(
            np.asarray([[0], [1]], dtype=int),
            field_name="start_roi_indices",
        )
