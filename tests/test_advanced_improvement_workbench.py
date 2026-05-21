"""Regression tests for advanced improvement manifest helpers."""

from __future__ import annotations

import json

from bayescatrack.experiments.advanced_improvement_workbench import (
    track2p_result_improvement_manifest,
)


def test_track2p_result_improvement_manifest_contains_key_variants(tmp_path):
    manifest = track2p_result_improvement_manifest(
        data_root="data",
        reference_root="reference",
        output_root="results",
        max_gap=3,
        transform_type="fov-affine",
    )
    manifest_path = tmp_path / "track2p_result_improvements.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert manifest["defaults"]["max_gap"] == 3
    assert manifest["defaults"]["transform_type"] == "fov-affine"
    assert manifest["defaults"]["include_non_cells"] is True

    run_names = {run["name"] for run in manifest["runs"]}
    assert "global-registered-iou-prior-sweep" in run_names
    assert "roi-aware-shifted-higher-order" in run_names
    assert "roi-aware-shifted-activity-tiebreaker" in run_names
    assert "calibrated-loso-local-evidence" in run_names
    assert "configurable-loso-local-evidence-hgb" in run_names
    assert "monotone-loso-local-evidence" in run_names
    assert "registration-qa" in run_names

    higher_order = next(
        run for run in manifest["runs"] if run["name"] == "roi-aware-shifted-higher-order"
    )
    assert higher_order["higher_order_triplet_weight"] > 0.0

    configurable = next(
        run for run in manifest["runs"] if run["name"] == "configurable-loso-local-evidence-hgb"
    )
    assert configurable["runner"] == "track2p-loso-calibration"
    assert "one_minus_weighted_dice" in configurable["feature_names"]
