from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.matching import build_track_rows_from_bundles, build_track_rows_from_matches


class _Bundle:
    def __init__(self, costs: object) -> None:
        self.pairwise_cost_matrix = np.asarray(costs, dtype=float)
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = np.array([0])
        self.measurement_roi_indices = np.array([1])


class _IndexValueError:
    def __index__(self) -> int:
        raise ValueError("synthetic broken index conversion")


class _IndexOverflow:
    def __index__(self) -> int:
        raise OverflowError("synthetic overflowing index conversion")


@pytest.mark.parametrize("bad_fill_value", [_IndexValueError(), _IndexOverflow()])
def test_build_track_rows_from_matches_normalizes_fill_value_index_errors(
    bad_fill_value: object,
) -> None:
    with pytest.raises(ValueError, match="fill_value"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.empty((0, 2), dtype=int)],
            start_roi_indices=np.array([0]),
            fill_value=bad_fill_value,
        )


@pytest.mark.parametrize("bad_fill_value", [_IndexValueError(), _IndexOverflow()])
def test_build_track_rows_from_bundles_normalizes_fill_value_index_errors(
    bad_fill_value: object,
) -> None:
    with pytest.raises(ValueError, match="fill_value"):
        build_track_rows_from_bundles(
            [_Bundle([[100.0]])],
            fill_value=bad_fill_value,
        )
