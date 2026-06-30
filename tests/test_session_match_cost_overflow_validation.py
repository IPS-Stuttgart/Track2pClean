from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from bayescatrack.matching import SessionMatchResult


def test_session_match_result_normalizes_oversized_fraction_cost():
    with pytest.raises(ValueError, match="finite numeric assignment costs"):
        SessionMatchResult(
            reference_session_name="day0",
            measurement_session_name="day1",
            reference_positions=np.asarray([0], dtype=int),
            measurement_positions=np.asarray([0], dtype=int),
            reference_roi_indices=np.asarray([10], dtype=int),
            measurement_roi_indices=np.asarray([20], dtype=int),
            costs=np.asarray([Fraction(10**400, 1)], dtype=object),
        )
