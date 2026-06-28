from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.analysis.growth import (
    _optional_roi,
    _target_sessions,
    _validate_session_index,
    affine_growth_summaries,
    radial_displacement_rows,
    radial_growth_summaries,
)
from bayescatrack.core.bridge import CalciumPlaneData, Track2pSession
from tests._support import run_module


def _plane_from_points(points_xy, roi_indices):
    width = max(int(point[0]) for point in points_xy) + 1
    height = max(int(point[1]) for point in points_xy) + 1
    masks = np.zeros((len(points_xy), height, width), dtype=bool)
    for roi_index, (x_coord, y_coord) in enumerate(points_xy):
        masks[roi_index, int(y_coord), int(x_coord)] = True
    return CalciumPlaneData(
        roi_masks=masks,
        roi_indices=np.asarray(roi_indices, dtype=int),
        source="test",
        plane_name="plane0",
    )


def _session(name, points_xy, roi_indices):
    return Track2pSession(
        session_dir=None,
        session_name=name,
        session_date=None,
        plane_data=_plane_from_points(points_xy, roi_indices),
    )


def test_radial_displacement_detects_outward_growth():
    source = [(3, 2), (2, 3), (1, 2), (2, 1)]
    target = [(4, 2), (2, 4), (0, 2), (2, 0)]
    sessions = [
        _session("s0", source, [10, 11, 12, 13]),
        _session("s1", target, [20, 21, 22, 23]),
    ]
    tracks = np.asarray(
        [
            [10, 20],
            [11, 21],
            [12, 22],
            [13, 23],
        ],
        dtype=object,
    )

    rows = radial_displacement_rows(sessions, tracks, center=(2, 2))
    summaries = radial_growth_summaries(rows)

    assert len(rows) == 4
    assert {row.radial_displacement for row in rows} == {1.0}
    assert summaries[0].outward_tracks == 4
    assert summaries[0].inward_tracks == 0
    assert summaries[0].mean_radial_displacement == pytest.approx(1.0)
    assert summaries[0].outward_sign_p_value == pytest.approx(1 / 16)


def test_affine_growth_summary_recovers_scale_and_translation():
    source = [(1, 1), (1, 3), (3, 1), (3, 3)]
    target = [(3, 1), (3, 5), (7, 1), (7, 5)]
    sessions = [
        _session("s0", source, [0, 1, 2, 3]),
        _session("s1", target, [10, 11, 12, 13]),
    ]
    tracks = np.asarray(
        [
            [0, 10],
            [1, 11],
            [2, 12],
            [3, 13],
        ],
        dtype=object,
    )

    summary = affine_growth_summaries(sessions, tracks)[0]

    assert summary.matrix_xx == pytest.approx(2.0)
    assert summary.matrix_yy == pytest.approx(2.0)
    assert summary.matrix_xy == pytest.approx(0.0)
    assert summary.matrix_yx == pytest.approx(0.0)
    assert summary.translation_x == pytest.approx(1.0)
    assert summary.translation_y == pytest.approx(-1.0)
    assert summary.determinant == pytest.approx(4.0)
    assert summary.isotropic_scale == pytest.approx(2.0)
    assert summary.residual_rmse == pytest.approx(0.0)


@pytest.mark.parametrize("bad_target_sessions", ["10", b"10", bytearray(b"10")])
def test_growth_target_sessions_rejects_string_like_sequences(bad_target_sessions):
    with pytest.raises(ValueError, match="target_sessions.*string-like"):
        _target_sessions(
            n_sessions=12,
            source_session=2,
            target_sessions=bad_target_sessions,
        )


def test_growth_optional_roi_rejects_fractional_values():
    assert _optional_roi(1) == 1
    assert _optional_roi(1.0) == 1
    assert _optional_roi("1.0") == 1
    with pytest.raises(ValueError, match="integer-like"):
        _optional_roi(1.5)
    with pytest.raises(ValueError, match="integer-like"):
        _optional_roi("1.5")
    assert _optional_roi("nan") is None


def test_growth_optional_roi_rejects_boolean_values():
    with pytest.raises(ValueError, match="boolean"):
        _optional_roi(True)
    with pytest.raises(ValueError, match="boolean"):
        _optional_roi(False)
    with pytest.raises(ValueError, match="boolean"):
        _optional_roi(np.bool_(True))


@pytest.mark.parametrize(
    "bad_index", [True, False, np.bool_(True), 1.5, "1.5", np.nan, np.inf]
)
def test_growth_session_index_rejects_bool_fractional_and_nonfinite_values(bad_index):
    with pytest.raises(ValueError, match="session index"):
        _validate_session_index(bad_index, 3)


@pytest.mark.parametrize("integer_like_index", [1, np.int64(1), 1.0, "1", "1.0"])
def test_growth_session_index_accepts_integer_like_values(integer_like_index):
    assert _validate_session_index(integer_like_index, 3) == 1


def test_growth_cli_help_is_registered():
    proc = run_module("-m", "bayescatrack", "growth", "--help")
    assert "radial" in proc.stdout
    assert "affine" in proc.stdout
