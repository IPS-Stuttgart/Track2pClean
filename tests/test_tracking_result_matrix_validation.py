"""Regression tests for tracking-result matrix validation."""

from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.registration import RegisteredConsecutiveBundles
from bayescatrack.tracking import SubjectTrackingResult


def _make_result(
    *,
    session_names=("s0", "s1"),
    track_rows=None,
    link_costs=None,
    link_target_indices=None,
    fill_value=-1,
) -> SubjectTrackingResult:
    if track_rows is None:
        track_rows = np.asarray([[0, 1]], dtype=int)
    track_rows_array = np.asarray(track_rows, dtype=object)
    if link_costs is None:
        link_costs = np.zeros(
            (track_rows_array.shape[0], max(track_rows_array.shape[1] - 1, 0)),
            dtype=float,
        )
    return SubjectTrackingResult(
        sessions=(),
        registered_bundles=RegisteredConsecutiveBundles(bundles=[]),
        match_results=(),
        session_names=tuple(session_names),
        track_rows=track_rows,
        link_costs=link_costs,
        link_target_indices=link_target_indices,
        fill_value=fill_value,
    )


@pytest.mark.parametrize(
    "track_rows",
    [
        np.asarray([[True, 1]], dtype=object),
        np.asarray([[0, 1.5]], dtype=object),
        np.asarray([[0, -2]], dtype=object),
    ],
)
def test_subject_tracking_result_rejects_malformed_track_rows(track_rows) -> None:
    with pytest.raises(ValueError, match="track_rows"):
        _make_result(track_rows=track_rows)


def test_subject_tracking_result_normalizes_integer_like_track_rows() -> None:
    result = _make_result(
        session_names=("s0", "s1", "s2"),
        track_rows=np.asarray([[0.0, np.int64(1), -1.0]], dtype=object),
        link_costs=np.asarray([[0.25, np.nan]], dtype=float),
    )

    np.testing.assert_array_equal(
        result.track_rows, np.asarray([[0, 1, -1]], dtype=int)
    )


@pytest.mark.parametrize(
    "link_target_indices",
    [
        np.asarray([[True, 2]], dtype=object),
        np.asarray([[1.5, 2]], dtype=object),
        np.asarray([[0, 2]], dtype=object),
        np.asarray([[1, 3]], dtype=object),
    ],
)
def test_subject_tracking_result_rejects_invalid_link_target_indices(
    link_target_indices,
) -> None:
    with pytest.raises(ValueError, match="link_target_indices"):
        _make_result(
            session_names=("s0", "s1", "s2"),
            track_rows=np.asarray([[0, 1, 2]], dtype=int),
            link_costs=np.asarray([[0.25, 0.5]], dtype=float),
            link_target_indices=link_target_indices,
        )


def test_subject_tracking_result_accepts_forward_link_target_indices() -> None:
    result = _make_result(
        session_names=("s0", "s1", "s2"),
        track_rows=np.asarray([[0, -1, 2]], dtype=int),
        link_costs=np.asarray([[0.25, np.nan]], dtype=float),
        link_target_indices=np.asarray([[2, -1.0]], dtype=object),
    )

    np.testing.assert_array_equal(
        result.link_target_indices,
        np.asarray([[2, -1]], dtype=int),
    )
