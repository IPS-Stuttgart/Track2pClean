from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.evaluation import normalize_track_matrix
from bayescatrack.evaluation.track2p_metrics import (
    normalize_track_matrix as normalize_track_matrix_facade,
)


def test_normalize_track_matrix_treats_vector_as_single_track_row():
    normalized = normalize_track_matrix([0, "10", None])

    assert normalized.shape == (1, 3)
    np.testing.assert_array_equal(normalized, np.asarray([[0, 10, -1]], dtype=int))


def test_track2p_metrics_facade_normalizes_vector_to_row_matrix():
    normalized = normalize_track_matrix_facade(np.asarray([0, -1, "nan"], dtype=object))

    assert normalized.shape == (1, 3)
    np.testing.assert_array_equal(normalized, np.asarray([[0, -1, -1]], dtype=int))


def test_normalize_track_matrix_rejects_scalar_input():
    with pytest.raises(ValueError, match="two-dimensional or a single track vector"):
        normalize_track_matrix(0)
