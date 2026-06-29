from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.multi_hypothesis import TrackHypothesis, hypotheses_to_matrix


def test_hypotheses_to_matrix_preserves_valid_equal_width_rows() -> None:
    matrix = hypotheses_to_matrix(
        [
            TrackHypothesis(row=(10, -1), cost=0.0),
            TrackHypothesis(row=(11, 12), cost=1.0),
        ]
    )

    assert np.issubdtype(matrix.dtype, np.integer)
    assert matrix.tolist() == [[10, -1], [11, 12]]


def test_hypotheses_to_matrix_rejects_inconsistent_widths() -> None:
    with pytest.raises(ValueError, match="same length"):
        hypotheses_to_matrix(
            [
                TrackHypothesis(row=(10, 11), cost=0.0),
                TrackHypothesis(row=(12,), cost=1.0),
            ]
        )


@pytest.mark.parametrize(
    "bad_value",
    [True, 1.5, "1", b"1", np.array([1])],
)
def test_hypotheses_to_matrix_rejects_silent_row_value_coercion(bad_value) -> None:
    with pytest.raises(ValueError, match=r"hypotheses\[0\]\.row\[0\].*integer"):
        hypotheses_to_matrix([TrackHypothesis(row=(bad_value, 2), cost=0.0)])


def test_hypotheses_to_matrix_rejects_text_row_container() -> None:
    with pytest.raises(ValueError, match="sequence of integer entries"):
        hypotheses_to_matrix([TrackHypothesis(row="12", cost=0.0)])
