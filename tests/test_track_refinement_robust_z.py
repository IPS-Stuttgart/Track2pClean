from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.track_refinement import (
    TrackSmoothingConfig,
    smoothed_track_positions,
    split_tracks_at_issues,
    track_geometry_issues,
)


def test_track_geometry_flags_isolated_outlier_when_mad_is_zero() -> None:
    """A single bad detection must not be hidden by the std fallback.

    With four nearly collinear detections and one swapped middle ROI, the global
    linear fit produces identical small residuals on the good points and one
    large residual on the swapped point.  The median absolute deviation is then
    zero; using the contaminated standard deviation as a fallback keeps the
    robust-z score below the default threshold even though the absolute residual
    is large enough for relinking/splitting.
    """

    rows = np.asarray([[0, 0, 0, 0, 0]], dtype=int)
    position_tables = tuple(
        {0: np.asarray([float(x), 0.0], dtype=float)}
        for x in (0.0, 10.0, 80.0, 30.0, 40.0)
    )

    issues = track_geometry_issues(
        rows,
        position_tables,
        config=TrackSmoothingConfig(),
    )

    assert [(issue.session_index, issue.roi_index) for issue in issues] == [(2, 0)]
    assert issues[0].residual == pytest.approx(48.0)
    assert np.isinf(issues[0].robust_z)


def test_track_geometry_keeps_clean_linear_tracks_unflagged() -> None:
    rows = np.asarray([[0, 0, 0, 0, 0]], dtype=int)
    position_tables = tuple(
        {0: np.asarray([10.0 * session_index, 0.0], dtype=float)}
        for session_index in range(5)
    )

    issues = track_geometry_issues(
        rows,
        position_tables,
        config=TrackSmoothingConfig(),
    )

    assert issues == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("residual_z_threshold", True),
        ("residual_z_threshold", float("nan")),
        ("residual_z_threshold", float("inf")),
        ("residual_z_threshold", 0.0),
        ("min_track_detections", False),
        ("min_track_detections", 1.5),
        ("min_track_detections", 1),
        ("min_edge_residual", True),
        ("min_edge_residual", float("nan")),
        ("min_edge_residual", float("inf")),
        ("min_edge_residual", -0.1),
        ("split_bad_edges", 1),
        ("fill_value", True),
        ("fill_value", 1.5),
    ],
)
def test_track_smoothing_config_rejects_invalid_controls(
    field: str, value: float | bool | int
) -> None:
    with pytest.raises(ValueError, match=field):
        TrackSmoothingConfig(**{field: value})


@pytest.mark.parametrize("fill_value", [True, 1.5])
def test_smoothed_track_positions_rejects_invalid_fill_value(
    fill_value: float | bool,
) -> None:
    with pytest.raises(ValueError, match="fill_value"):
        smoothed_track_positions(
            [[0]], ({0: np.asarray([0.0, 0.0])},), fill_value=fill_value
        )


@pytest.mark.parametrize("fill_value", [True, 1.5])
def test_split_tracks_at_issues_rejects_invalid_fill_value(
    fill_value: float | bool,
) -> None:
    with pytest.raises(ValueError, match="fill_value"):
        split_tracks_at_issues([[0]], (), fill_value=fill_value)
