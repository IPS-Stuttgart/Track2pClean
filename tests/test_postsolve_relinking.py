from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.postsolve_relinking import (
    PostSolveRelinkingConfig,
    relink_tracks_at_geometry_issues,
)
from bayescatrack.association.track_refinement import TrackGeometryIssue


def _issue() -> TrackGeometryIssue:
    return TrackGeometryIssue(
        track_index=0,
        session_index=1,
        roi_index=21,
        residual=9.0,
        robust_z=4.0,
        suggested_action="split_or_relink",
    )


def test_relink_uses_outgoing_edge_when_available():
    rows = np.array([[10, 21, 30]], dtype=int)
    roi_indices_by_session = ([10], [20, 21, 22], [30])
    pairwise_costs = {
        (0, 1): np.array([[0.1, 0.2, 0.4]], dtype=float),
        (1, 2): np.array([[10.0], [8.0], [0.1]], dtype=float),
    }

    relinked = relink_tracks_at_geometry_issues(
        rows,
        [_issue()],
        pairwise_costs,
        roi_indices_by_session=roi_indices_by_session,
        config=PostSolveRelinkingConfig(
            max_edge_cost=None,
            min_cost_improvement=0.0,
        ),
    )

    assert relinked.tolist() == [[10, 22, 30]]


def test_relink_can_disable_outgoing_edge_evidence_for_legacy_behavior():
    rows = np.array([[10, 21, 30]], dtype=int)
    roi_indices_by_session = ([10], [20, 21, 22], [30])
    pairwise_costs = {
        (0, 1): np.array([[0.1, 0.2, 0.4]], dtype=float),
        (1, 2): np.array([[10.0], [8.0], [0.1]], dtype=float),
    }

    relinked = relink_tracks_at_geometry_issues(
        rows,
        [_issue()],
        pairwise_costs,
        roi_indices_by_session=roi_indices_by_session,
        config=PostSolveRelinkingConfig(
            max_edge_cost=None,
            min_cost_improvement=0.0,
            bidirectional_next_weight=0.0,
        ),
    )

    assert relinked.tolist() == [[10, 20, 30]]


def test_relink_rejects_nested_roi_index_vectors():
    rows = np.array([[10, 21]], dtype=int)

    with pytest.raises(
        ValueError,
        match=r"roi_indices_by_session\[1\].*one-dimensional",
    ):
        relink_tracks_at_geometry_issues(
            rows,
            [_issue()],
            {(0, 1): np.array([[0.1, 0.2]], dtype=float)},
            roi_indices_by_session=([10], [[20], [21]]),
            config=PostSolveRelinkingConfig(max_edge_cost=None),
        )


def test_relink_rejects_duplicate_roi_index_vectors():
    rows = np.array([[10, 21]], dtype=int)

    with pytest.raises(
        ValueError,
        match=r"roi_indices_by_session\[1\].*unique ROI indices",
    ):
        relink_tracks_at_geometry_issues(
            rows,
            [_issue()],
            {(0, 1): np.array([[0.1, 0.2]], dtype=float)},
            roi_indices_by_session=([10], [21, 21]),
            config=PostSolveRelinkingConfig(max_edge_cost=None),
        )


@pytest.mark.parametrize(
    "bad_roi_indices",
    [
        [20.5, 21],
        [True, 21],
        [np.bool_(False), 21],
        [-1, 21],
        ["20", 21],
        [np.asarray(20), 21],
        [np.asarray(-1), 21],
        [np.asarray([20]), 21],
    ],
)
def test_relink_rejects_non_integral_roi_index_vectors(bad_roi_indices):
    rows = np.array([[10, 21]], dtype=int)

    with pytest.raises(ValueError, match=r"roi_indices_by_session\[1\].*ROI indices"):
        relink_tracks_at_geometry_issues(
            rows,
            [_issue()],
            {(0, 1): np.array([[0.1, 0.2]], dtype=float)},
            roi_indices_by_session=([10], bad_roi_indices),
            config=PostSolveRelinkingConfig(max_edge_cost=None),
        )


def test_relink_rejects_non_matrix_pairwise_costs():
    rows = np.array([[10, 21]], dtype=int)

    with pytest.raises(ValueError, match=r"pairwise_costs.*two-dimensional"):
        relink_tracks_at_geometry_issues(
            rows,
            [_issue()],
            {(0, 1): np.array([0.1, 0.2], dtype=float)},
            roi_indices_by_session=([10], [20, 21]),
            config=PostSolveRelinkingConfig(max_edge_cost=None),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_edge_cost", True),
        ("max_edge_cost", float("nan")),
        ("max_edge_cost", float("inf")),
        ("max_edge_cost", -0.1),
        ("min_cost_improvement", False),
        ("min_cost_improvement", float("nan")),
        ("min_cost_improvement", float("inf")),
        ("min_cost_improvement", -0.1),
        ("enforce_unique_session_rois", 1),
        ("fill_value", True),
        ("fill_value", 0),
        ("fill_value", np.int64(0)),
        ("fill_value", 1.5),
        ("bidirectional_next_weight", True),
        ("bidirectional_next_weight", float("nan")),
        ("bidirectional_next_weight", float("inf")),
        ("bidirectional_next_weight", -0.1),
    ],
)
def test_postsolve_relinking_config_rejects_invalid_controls(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        PostSolveRelinkingConfig(**{field: value})


def test_postsolve_relinking_accepts_negative_fill_sentinel() -> None:
    assert PostSolveRelinkingConfig(fill_value=-2).fill_value == -2
