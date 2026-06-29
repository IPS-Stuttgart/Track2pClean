from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.matching import (
    build_track_rows_from_bundles,
    build_track_rows_from_matches,
)


class _Bundle:
    def __init__(self, costs: object) -> None:
        self.pairwise_cost_matrix = np.asarray(costs, dtype=float)
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = np.array([0])
        self.measurement_roi_indices = np.array([1])


class _BadIndexValueError:
    def __index__(self) -> int:
        raise ValueError("bad index")


class _BadIndexOverflowError:
    def __index__(self) -> int:
        raise OverflowError("too large")


def test_build_track_rows_from_matches_rejects_string_start_session_index() -> None:
    with pytest.raises(ValueError, match="start_session_index must be an integer"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1]], dtype=int)],
            start_roi_indices=np.array([0]),
            start_session_index="1",  # type: ignore[arg-type]
        )


def test_build_track_rows_from_matches_rejects_string_fill_value() -> None:
    with pytest.raises(
        ValueError, match="fill_value must be a negative integer sentinel"
    ):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.empty((0, 2), dtype=int)],
            start_roi_indices=np.array([0]),
            fill_value="-1",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "fill_value",
    [_BadIndexValueError(), _BadIndexOverflowError()],
)
def test_build_track_rows_from_matches_normalizes_bad_index_fill_value(
    fill_value: object,
) -> None:
    with pytest.raises(ValueError, match="fill_value must be an integer"):
        build_track_rows_from_matches(
            ("s1",),
            [],
            start_roi_indices=np.array([0]),
            fill_value=fill_value,  # type: ignore[arg-type]
        )


def test_build_track_rows_from_bundles_rejects_string_fill_value() -> None:
    with pytest.raises(
        ValueError, match="fill_value must be a negative integer sentinel"
    ):
        build_track_rows_from_bundles(
            [_Bundle([[100.0]])],
            fill_value="-1",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "fill_value",
    [_BadIndexValueError(), _BadIndexOverflowError()],
)
def test_build_track_rows_from_bundles_normalizes_bad_index_fill_value(
    fill_value: object,
) -> None:
    with pytest.raises(ValueError, match="fill_value must be an integer"):
        build_track_rows_from_bundles(
            [_Bundle([[100.0]])],
            fill_value=fill_value,  # type: ignore[arg-type]
        )
