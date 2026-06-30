from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.track_refinement import (
    TrackSmoothingConfig,
    track_geometry_issues,
)


class _OverflowingIndex:
    def __index__(self) -> int:
        raise OverflowError("synthetic index overflow")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"residual_z_threshold": np.nan},
            "residual_z_threshold must be a finite positive value",
        ),
        (
            {"residual_z_threshold": True},
            "residual_z_threshold must be a finite positive value",
        ),
        (
            {"residual_z_threshold": np.array([3.5])},
            "residual_z_threshold must be a finite positive value",
        ),
        (
            {"residual_z_threshold": np.array([True])},
            "residual_z_threshold must be a finite positive value",
        ),
        (
            {"min_track_detections": 2.5},
            "min_track_detections must be an integer",
        ),
        (
            {"min_track_detections": True},
            "min_track_detections must be an integer",
        ),
        (
            {"min_track_detections": 1},
            "min_track_detections must be at least 2",
        ),
        (
            {"min_edge_residual": np.inf},
            "min_edge_residual must be a finite non-negative value",
        ),
        (
            {"min_edge_residual": True},
            "min_edge_residual must be a finite non-negative value",
        ),
        (
            {"min_edge_residual": [0.0]},
            "min_edge_residual must be a finite non-negative value",
        ),
        (
            {"min_edge_residual": np.array([False])},
            "min_edge_residual must be a finite non-negative value",
        ),
        ({"split_bad_edges": 1}, "split_bad_edges must be a boolean"),
        ({"fill_value": 0.5}, "fill_value must be a negative integer sentinel"),
        ({"fill_value": False}, "fill_value must be a negative integer sentinel"),
    ],
)
def test_track_smoothing_config_rejects_silent_control_coercions(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrackSmoothingConfig(**kwargs)


def test_track_smoothing_config_wraps_min_detection_index_errors() -> None:
    with pytest.raises(ValueError, match="min_track_detections must be an integer"):
        TrackSmoothingConfig(min_track_detections=_OverflowingIndex())


def test_track_smoothing_config_normalizes_integer_like_controls() -> None:
    config = TrackSmoothingConfig(
        residual_z_threshold="3.5",
        min_track_detections=np.int64(4),
        min_edge_residual="0.0",
        split_bad_edges=np.bool_(False),
        fill_value=np.int64(-1),
    )

    assert config.residual_z_threshold == pytest.approx(3.5)
    assert config.min_track_detections == 4
    assert config.min_edge_residual == pytest.approx(0.0)
    assert config.split_bad_edges is False
    assert config.fill_value == -1


def test_track_smoothing_config_accepts_numpy_scalar_float_controls() -> None:
    config = TrackSmoothingConfig(
        residual_z_threshold=np.array(3.5),
        min_edge_residual=np.array(0.0),
    )

    assert config.residual_z_threshold == pytest.approx(3.5)
    assert config.min_edge_residual == pytest.approx(0.0)


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
