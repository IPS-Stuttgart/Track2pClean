"""Regression tests for global tracking link-edge validation."""

from __future__ import annotations

import types

import numpy as np
import pytest

import bayescatrack.tracking as tracking
from bayescatrack.tracking import SubjectTrackingResult


class _Plane:
    n_rois = 1
    roi_indices = np.asarray([0], dtype=int)


class _Session:
    def __init__(self, name: str) -> None:
        self.session_name = name
        self.plane_data = _Plane()


def _make_tracking_result(*, global_link_edges, global_link_costs=None) -> SubjectTrackingResult:
    if global_link_costs is None:
        global_link_costs = np.zeros((1, len(tuple(global_link_edges))), dtype=float)
    return SubjectTrackingResult(
        sessions=(),
        registered_bundles=None,
        match_results=(),
        session_names=("s0", "s1"),
        track_rows=np.asarray([[0, 1]], dtype=int),
        link_costs=np.asarray([[0.1]], dtype=float),
        global_link_edges=global_link_edges,
        global_link_costs=global_link_costs,
    )


@pytest.mark.parametrize(
    "global_link_edges",
    [
        ((True, 1),),
        ((0, np.array(1.5)),),
        ((1, 0),),
        ((0, 2),),
        ((0, 1), (0, 1)),
    ],
)
def test_subject_tracking_result_rejects_invalid_global_link_edges(global_link_edges) -> None:
    with pytest.raises(ValueError, match="global_link_edges"):
        _make_tracking_result(global_link_edges=global_link_edges)


def test_subject_tracking_result_normalizes_integer_like_global_link_edges() -> None:
    result = _make_tracking_result(global_link_edges=((np.int64(0), np.float64(1.0)),))

    assert result.global_link_edges == ((0, 1),)


def test_build_global_link_cost_matrices_rejects_missing_edge_costs() -> None:
    sessions = (_Session("s0"), _Session("s1"), _Session("s2"))
    global_assignment = types.SimpleNamespace(
        session_edges=((0, 1), (0, 2)),
        pairwise_costs={(0, 1): np.asarray([[0.1]], dtype=float)},
    )

    with pytest.raises(ValueError, match="missing from pairwise_costs"):
        tracking._build_global_link_cost_matrices(  # pylint: disable=protected-access
            global_assignment,
            sessions,
            np.asarray([[0, 0, 0]], dtype=int),
            fallback_match_results=(),
            fill_value=-1,
        )
