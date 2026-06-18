"""Regression tests for advanced improvement manifest helpers."""

from __future__ import annotations

import json

import pytest
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
    assert "track2p-policy-coherence-suffix-teacher-rescue" in run_name_set
    assert "track2p-policy-growth-veto-cleanup" in run_name_set
    assert "track2p-policy-coherence-suffix-growth-veto-cleanup" in run_name_set
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

    suffix_teacher_rescue = next(
        run
        for run in manifest["runs"]
        if run["name"] == "track2p-policy-coherence-suffix-teacher-rescue"
    )
    assert (
        suffix_teacher_rescue["runner"]
        == "track2p-policy-coherence-suffix-teacher-rescue"
    )
    assert suffix_teacher_rescue["suffix_path_length"] == 2
    assert suffix_teacher_rescue["min_shifted_iou"] == 0.3
    assert suffix_teacher_rescue["teacher_edge_order"] == "structural"
    assert suffix_teacher_rescue["teacher_action_filter"] == "all"
    assert suffix_teacher_rescue["teacher_feature_preset"] == "none"
    assert suffix_teacher_rescue["max_applied_teacher_edits"] == -1

    growth_veto_cleanup = next(
        run
        for run in manifest["runs"]
        if run["name"] == "track2p-policy-growth-veto-cleanup"
    )
    assert growth_veto_cleanup["runner"] == "track2p-policy-growth-veto-cleanup"
    assert growth_veto_cleanup["suffix_path_length"] == 2
    assert "teacher_edge_order" not in growth_veto_cleanup
    assert "teacher_action_filter" not in growth_veto_cleanup
    assert "teacher_feature_preset" not in growth_veto_cleanup
    assert "max_applied_teacher_edits" not in growth_veto_cleanup
    assert growth_veto_cleanup["anchor_min_registered_iou"] == 0.5
    assert growth_veto_cleanup["anchor_min_shifted_iou"] == 0.3
    assert growth_veto_cleanup["anchor_min_cell_probability"] == 0.8
    assert growth_veto_cleanup["min_growth_residual_mahalanobis"] == 20.0
    assert growth_veto_cleanup["min_veto_registered_iou"] == 0.45
    assert growth_veto_cleanup["min_veto_shifted_iou"] == 0.6
    assert growth_veto_cleanup["max_veto_registered_iou"] == 0.6
    assert growth_veto_cleanup["max_veto_shifted_iou"] == 0.8
    assert growth_veto_cleanup["max_veto_min_cell_probability"] == 0.65

    suffix_growth_veto_cleanup = next(
        run
        for run in manifest["runs"]
        if run["name"] == "track2p-policy-coherence-suffix-growth-veto-cleanup"
    )
    assert (
        suffix_growth_veto_cleanup["runner"]
        == "track2p-policy-coherence-suffix-growth-veto-cleanup"
    )
    assert "growth_veto_base" not in suffix_growth_veto_cleanup
    assert suffix_growth_veto_cleanup["max_veto_local_neighbor_distortion"] is None
    assert "teacher_edge_order" not in suffix_growth_veto_cleanup
    assert suffix_growth_veto_cleanup["max_vetoes_per_subject"] == 1

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


@pytest.mark.parametrize(
    "max_gap",
    [True, False, 0, -1, 1.5, "2", float("nan"), float("inf")],
)
def test_track2p_result_improvement_manifest_rejects_invalid_max_gap(
    max_gap: object,
) -> None:
    with pytest.raises(ValueError, match="max_gap"):
        track2p_result_improvement_manifest(
            data_root="data",
            reference_root="reference",
            output_root="results",
            max_gap=max_gap,  # type: ignore[arg-type]
        )
