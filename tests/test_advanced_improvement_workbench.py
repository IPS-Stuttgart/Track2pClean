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

    run_names = [run["name"] for run in manifest["runs"]]
    assert len(run_names) == len(set(run_names))
    run_name_set = set(run_names)
    assert "track2p-policy" in run_name_set
    assert "track2p-policy-component-cleanup" in run_name_set
    assert "track2p-policy-teacher-adjacent-rescue" in run_name_set
    assert "track2p-policy-dp" in run_name_set
    assert "track2p-policy-pruned" in run_name_set
    assert "global-registered-iou-prior-sweep" in run_name_set
    assert "roi-aware-shifted-higher-order" in run_names
    assert "roi-aware-shifted-activity-tiebreaker" in run_names
    assert "calibrated-loso-local-evidence" in run_names
    assert "configurable-loso-local-evidence-hgb" in run_names
    assert "monotone-loso-local-evidence" in run_names
    assert "registration-qa" in run_names

    higher_order = next(
        run
        for run in manifest["runs"]
        if run["name"] == "roi-aware-shifted-higher-order"
    )
    assert higher_order["higher_order_triplet_weight"] > 0.0

    configurable = next(
        run
        for run in manifest["runs"]
        if run["name"] == "configurable-loso-local-evidence-hgb"
    )
    assert configurable["runner"] == "track2p-loso-calibration"
    assert "one_minus_weighted_dice" in configurable["feature_names"]

    component_cleanup = next(
        run
        for run in manifest["runs"]
        if run["name"] == "track2p-policy-component-cleanup"
    )
    assert component_cleanup["runner"] == "track2p-policy-component-audit"
    assert component_cleanup["apply_splits"] is True
    assert component_cleanup["split_risk_threshold"] == 1.5
    assert component_cleanup["min_side_observations"] == 2

    teacher_rescue = next(
        run
        for run in manifest["runs"]
        if run["name"] == "track2p-policy-teacher-adjacent-rescue"
    )
    assert teacher_rescue["runner"] == "track2p-policy-teacher-adjacent-rescue"
    assert teacher_rescue["allow_completing_rescue"] is False
    assert teacher_rescue["allow_source_backfill"] is True
    assert teacher_rescue["allow_seed_source_backfill"] is False
    assert teacher_rescue["allow_completing_seed_source_backfill"] is False
    assert teacher_rescue["allow_fragment_merges"] is True

    seed_source_rescue = next(
        run
        for run in manifest["runs"]
        if run["name"] == "track2p-policy-teacher-adjacent-rescue-seed-source"
    )
    assert seed_source_rescue["allow_seed_source_backfill"] is True


def test_track2p_result_improvement_manifest_adds_experimental_policy_dp_once():
    manifest = track2p_result_improvement_manifest(
        data_root="data",
        reference_root="reference",
        output_root="results",
        include_experimental_policy_dp=True,
    )

    run_names = [run["name"] for run in manifest["runs"]]
    assert len(run_names) == len(set(run_names))
    assert "track2p-policy-dp" in run_names
    assert "track2p-policy-dp-experimental" in run_names

    experimental = next(
        run
        for run in manifest["runs"]
        if run["name"] == "track2p-policy-dp-experimental"
    )
    assert experimental["row_top_k"] == 3
    assert experimental["path_selection_beam_width"] == 512
