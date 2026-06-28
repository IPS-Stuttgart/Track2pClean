from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData, load_suite2p_plane, load_track2p_subject
from bayescatrack.track2p_registration import register_plane_pair


def _write_raw_npy_plane(plane_dir, *, n_rois: int = 1) -> None:
    plane_dir.mkdir(parents=True, exist_ok=True)
    rois = np.zeros((n_rois, 4, 5), dtype=bool)
    for roi_index in range(n_rois):
        rois[roi_index, 1 + roi_index % 2, 2 + roi_index % 2] = True
    np.save(plane_dir / "rois.npy", rois)
    np.save(plane_dir / "F.npy", np.ones((n_rois, 3), dtype=float))
    np.save(plane_dir / "fov.npy", rois.astype(float).sum(axis=0))


def test_load_suite2p_plane_rejects_negative_roi_coordinates(tmp_path):
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([-1, 1], dtype=int),
                "xpix": np.asarray([0, 1], dtype=int),
                "lam": np.asarray([1.0, 1.0], dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)
    np.save(tmp_path / "ops.npy", {"Ly": 4, "Lx": 5})

    with pytest.raises(
        ValueError, match="Suite2p ROI ypix pixel coordinates must be non-negative"
    ):
        load_suite2p_plane(tmp_path, load_traces=False, load_spike_traces=False)


def test_load_suite2p_plane_rejects_lam_shape_mismatch(tmp_path):
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([0, 1], dtype=int),
                "xpix": np.asarray([0, 1], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)
    np.save(tmp_path / "ops.npy", {"Ly": 4, "Lx": 5})

    with pytest.raises(ValueError, match="lam shape"):
        load_suite2p_plane(tmp_path, load_traces=False, load_spike_traces=False)


def test_auto_input_format_falls_back_from_broken_suite2p_to_raw_npy(tmp_path):
    subject_dir = tmp_path / "subject"
    session_dir = subject_dir / "2024-01-01_session"
    (session_dir / "suite2p" / "plane0").mkdir(parents=True)
    _write_raw_npy_plane(session_dir / "data_npy" / "plane0")

    sessions = load_track2p_subject(
        subject_dir, input_format="auto", plane_name="plane0"
    )

    assert len(sessions) == 1
    assert sessions[0].plane_data.source == "raw_npy"
    assert sessions[0].plane_data.roi_masks.shape == (1, 4, 5)


def test_auto_input_format_reports_all_existing_candidate_failures(tmp_path):
    subject_dir = tmp_path / "subject"
    session_dir = subject_dir / "2024-01-01_session"
    (session_dir / "suite2p" / "plane0").mkdir(parents=True)
    (session_dir / "data_npy" / "plane0").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Could not load any auto input format"):
        load_track2p_subject(subject_dir, input_format="auto", plane_name="plane0")


def test_explicit_registration_options_are_not_silently_ignored():
    fov = np.zeros((16, 16), dtype=float)
    fov[5:9, 6:10] = 1.0
    masks = fov[None, :, :] > 0.0
    reference = CalciumPlaneData(masks, fov=fov, source="reference")
    measurement = CalciumPlaneData(masks, fov=fov, source="measurement")

    with pytest.raises(TypeError, match="unexpected_option"):
        register_plane_pair(
            reference,
            measurement,
            transform_type="fov-affine",
            registration_options={"unexpected_option": True},
        )


def test_fov_affine_registration_options_reach_backend():
    fov = np.zeros((40, 40), dtype=float)
    fov[8:12, 8:12] = 1.0
    fov[24:28, 25:29] = 1.0
    masks = fov[None, :, :] > 0.0
    reference = CalciumPlaneData(masks, fov=fov, source="reference")
    measurement = CalciumPlaneData(masks, fov=fov, source="measurement")

    registered = register_plane_pair(
        reference,
        measurement,
        transform_type="fov-affine",
        registration_options={"grid_shape": (2, 2), "min_tile_size": 8},
    )

    assert registered.ops is not None
    assert registered.ops["registration_backend"] == "fov-affine"
    assert registered.ops["fov_affine_tile_count"] <= 4
