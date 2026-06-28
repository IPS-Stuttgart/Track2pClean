from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import export_subject_to_npz


def _write_minimal_raw_npy_plane(plane_dir):
    plane_dir.mkdir(parents=True)
    roi_masks = np.zeros((1, 2, 2), dtype=bool)
    roi_masks[0, 0, 1] = True
    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(plane_dir / "F.npy", np.array([[1.0, 2.0, 3.0]], dtype=float))
    np.save(plane_dir / "fov.npy", np.ones((2, 2), dtype=float))


@pytest.mark.parametrize(
    "flag_name",
    ["include_behavior", "include_masks", "weighted", "validate_pyrecest"],
)
def test_export_subject_to_npz_rejects_string_boolean_controls(tmp_path, flag_name):
    subject_dir = tmp_path / "jm123"
    _write_minimal_raw_npy_plane(subject_dir / "2024-05-01_a" / "data_npy" / "plane0")

    with pytest.raises(ValueError, match=f"{flag_name} must be a boolean"):
        export_subject_to_npz(
            subject_dir,
            tmp_path / "subject.npz",
            **{flag_name: "false"},
        )


def test_export_subject_to_npz_accepts_numpy_boolean_controls(tmp_path):
    subject_dir = tmp_path / "jm123"
    _write_minimal_raw_npy_plane(subject_dir / "2024-05-01_a" / "data_npy" / "plane0")

    summary = export_subject_to_npz(
        subject_dir,
        tmp_path / "subject.npz",
        include_behavior=np.bool_(False),
        include_masks=np.bool_(False),
        weighted=np.bool_(False),
        validate_pyrecest=np.bool_(False),
    )

    assert summary["n_sessions"] == 1
