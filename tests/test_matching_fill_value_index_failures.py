from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.matching import (
    build_track_rows_from_bundles,
    build_track_rows_from_matches,
)


class _Bundle:
    pairwise_cost_matrix = np.array([[100.0]])
    reference_session_name = "s1"
    measurement_session_name = "s2"
    reference_roi_indices = np.array([0])
    measurement_roi_indices = np.array([1])


class _IndexValueError:
    def __index__(self) -> int:
        raise ValueError("invalid")


class _IndexOverflowError:
    def __index__(self) -> int:
        raise OverflowError("invalid")


@pytest.mark.parametrize("fill_value", [_IndexValueError(), _IndexOverflowError()])
def test_build_track_rows_from_matches_normalizes_index_failures(
    fill_value: object,
) -> None:
    with pytest.raises(ValueError, match="fill_value must be an integer"):
        build_track_rows_from_matches(
            ("s1",),
            [],
            start_roi_indices=np.array([0]),
            fill_value=fill_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("fill_value", [_IndexValueError(), _IndexOverflowError()])
def test_build_track_rows_from_bundles_normalizes_index_failures(
    fill_value: object,
) -> None:
    with pytest.raises(ValueError, match="fill_value must be an integer"):
        build_track_rows_from_bundles(
            [_Bundle()],
            fill_value=fill_value,  # type: ignore[arg-type]
        )
