from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.cli import _handle_benchmark
from bayescatrack.experiments.nonrigid_backend_audit import (
    format_nonrigid_registration_backend_audit_table,
    summarize_registration_backend_audit_edges,
)
from bayescatrack.experiments import nonrigid_backend_audit


def test_nonrigid_backend_audit_surfaces_dense_warp_diagnostics():
    rows = [
        {
            "cost": "registered-iou",
            "subject": "jm001",
            "registration_backend": "bayescatrack-nonrigid",
            "transform_type": "tps",
            "registered_plane_source": "raw_npy_tps_registered",
            "registration_backend_reason": "image-driven dense inverse FOV warp",
            "gt_link_rows": 5,
            "fov_translation_shift_y": np.nan,
            "fov_translation_shift_x": np.nan,
            "fov_translation_peak_correlation": np.nan,
            "nonrigid_registration_backend": "thin-plate-spline-landmark-warp",
            "nonrigid_registration_grid_shape": "5x5",
            "nonrigid_registration_landmarks": 25,
            "nonrigid_registration_fit_rmse": 1.5,
            "nonrigid_registration_inverse_warp_valid_fraction": 0.95,
            "nonrigid_registration_fallback_translation": False,
            "nonrigid_registration_tps_regularization": 1.0e-3,
            "nonrigid_registration_optical_flow_iterations": 12,
            "nonrigid_registration_optical_flow_alpha": 25.0,
        },
        {
            "cost": "registered-iou",
            "subject": "jm002",
            "registration_backend": "bayescatrack-nonrigid",
            "transform_type": "tps",
            "registered_plane_source": "raw_npy_tps_registered",
            "registration_backend_reason": "image-driven dense inverse FOV warp",
            "gt_link_rows": 7,
            "fov_translation_shift_y": np.nan,
            "fov_translation_shift_x": np.nan,
            "fov_translation_peak_correlation": np.nan,
            "nonrigid_registration_backend": "thin-plate-spline-landmark-warp",
            "nonrigid_registration_grid_shape": "5x5",
            "nonrigid_registration_landmarks": 9,
            "nonrigid_registration_fit_rmse": 2.5,
            "nonrigid_registration_inverse_warp_valid_fraction": 0.85,
            "nonrigid_registration_fallback_translation": True,
            "nonrigid_registration_tps_regularization": 1.0e-3,
            "nonrigid_registration_optical_flow_iterations": 12,
            "nonrigid_registration_optical_flow_alpha": 25.0,
        },
    ]

    audit_rows = summarize_registration_backend_audit_edges(rows)

    assert len(audit_rows) == 1
    row = audit_rows[0]
    assert row["registration_backend"] == "bayescatrack-nonrigid"
    assert row["nonrigid_registration_backend"] == ("thin-plate-spline-landmark-warp")
    assert row["nonrigid_registration_grid_shape"] == "5x5"
    assert row["edge_count"] == 2
    assert row["gt_link_rows"] == 12
    assert row["subject_count"] == 2
    assert row["median_nonrigid_registration_landmarks"] == pytest.approx(17.0)
    assert row["median_nonrigid_registration_fit_rmse"] == pytest.approx(2.0)
    assert row[
        "median_nonrigid_registration_inverse_warp_valid_fraction"
    ] == pytest.approx(0.9)
    assert row["nonrigid_registration_fallback_translation_rate"] == pytest.approx(0.5)
    assert row["median_nonrigid_registration_tps_regularization"] == pytest.approx(
        1.0e-3
    )
    assert row["median_nonrigid_registration_optical_flow_iterations"] == pytest.approx(
        12.0
    )
    assert row["median_nonrigid_registration_optical_flow_alpha"] == pytest.approx(25.0)

    table = format_nonrigid_registration_backend_audit_table(audit_rows)
    assert "median_nonrigid_registration_fit_rmse" in table
    assert "thin-plate-spline-landmark-warp" in table


def test_benchmark_help_lists_nonrigid_backend_audit(capsys):
    assert _handle_benchmark(["--help"]) == 0

    captured = capsys.readouterr()
    assert "nonrigid-backend-audit" in captured.out


def test_nonrigid_backend_audit_cli_writes_csv(tmp_path, monkeypatch):
    rows = [
        {
            "cost": "registered-iou",
            "registration_backend": "bayescatrack-nonrigid",
            "transform_type": "tps",
            "registered_plane_source": "raw_npy_tps_registered",
            "edge_count": 1,
            "gt_link_rows": 3,
            "subject_count": 1,
            "subjects": "jm001",
            "nonrigid_registration_backend": "thin-plate-spline-landmark-warp",
            "median_nonrigid_registration_fit_rmse": 1.25,
        }
    ]

    def fake_run_registration_backend_audit_report(config):
        assert config.data == tmp_path
        assert config.transform_type == "tps"
        return rows

    monkeypatch.setattr(
        nonrigid_backend_audit,
        "run_registration_backend_audit_report",
        fake_run_registration_backend_audit_report,
    )
    output_path = tmp_path / "nonrigid_backend_audit.csv"

    assert nonrigid_backend_audit.main(
        ["--data", str(tmp_path), "--transform-type", "tps", "--output", str(output_path), "--format", "csv", "--no-progress"]
    ) == 0
    assert "median_nonrigid_registration_fit_rmse" in output_path.read_text()
    assert "thin-plate-spline-landmark-warp" in output_path.read_text()
