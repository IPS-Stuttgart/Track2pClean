from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests import _support  # noqa: F401
from bayescatrack.matching import build_track_rows_from_bundles, build_track_rows_from_matches


_SEQUENCE_ERROR = "not a bare string"
_DUPLICATE_ERROR = "unique session names"


def test_build_track_rows_from_matches_rejects_bare_session_names_string():
    with pytest.raises(ValueError, match=_SEQUENCE_ERROR):
        build_track_rows_from_matches(
            "s0",
            [{}],
            start_roi_indices=[0],
        )


def test_build_track_rows_from_matches_rejects_duplicate_session_names():
    with pytest.raises(ValueError, match=_DUPLICATE_ERROR):
        build_track_rows_from_matches(
            ["session0", "session0"],
            [{}],
            start_roi_indices=[0],
        )


def test_build_track_rows_from_matches_accepts_unique_session_names():
    rows = build_track_rows_from_matches(
        ["session0", "session1"],
        [{}],
        start_roi_indices=[7],
    )

    assert rows.tolist() == [[7, -1]]


def test_build_track_rows_from_bundles_rejects_duplicate_derived_session_names():
    bundle = SimpleNamespace(
        reference_session_name="session0",
        measurement_session_name="session0",
        pairwise_cost_matrix=np.zeros((1, 1), dtype=float),
        reference_roi_indices=np.array([0], dtype=int),
        measurement_roi_indices=np.array([0], dtype=int),
    )

    with pytest.raises(ValueError, match=_DUPLICATE_ERROR):
        build_track_rows_from_bundles([bundle])
