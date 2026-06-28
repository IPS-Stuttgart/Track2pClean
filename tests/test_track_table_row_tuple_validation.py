import numpy as np
import pytest

from bayescatrack.ground_truth_eval import TrackTable


def _table() -> TrackTable:
    return TrackTable(
        session_names=("s1", "s2", "s3"),
        tracks=np.array([[1, 2, 3], [4, -1, 6]], dtype=int),
    )


def test_row_tuples_rejects_boolean_horizon():
    with pytest.raises(ValueError, match="horizon"):
        _table().row_tuples(horizon=True)


def test_row_tuples_rejects_non_boolean_require_complete():
    with pytest.raises(ValueError, match="require_complete"):
        _table().row_tuples(require_complete=1)


def test_row_tuples_accepts_numpy_scalar_controls():
    assert _table().row_tuples(
        horizon=np.int64(2),
        require_complete=np.bool_(False),
    ) == [(1, 2), (4, -1)]
