from __future__ import annotations

from bayescatrack.experiments._teacher_rescue_repair_preset_manifest_integration import (
    _expand_teacher_repair_preset,
)


def test_missing_seed_repair_preset_expands_to_narrow_defaults():
    options = _expand_teacher_repair_preset(
        {"teacher_repair_preset": "missing-seed-high-confidence"}
    )

    assert options["allow_source_backfill"] is False
    assert options["allow_seed_source_backfill"] is True
    assert options["allow_completing_seed_source_backfill"] is True
    assert options["teacher_edge_order"] == "dynamic-seed-confidence"
    assert options["teacher_action_filter"] == "seed-source-backfill"
    assert options["teacher_feature_preset"] == "seed-source-high-confidence"
    assert options["min_component_observations"] == 2
    assert options["max_applied_edits"] == 2


def test_missing_seed_cell_confident_preset_expands_to_seed_source_gate():
    options = _expand_teacher_repair_preset(
        {"teacher_repair_preset": "missing-seed-cell-confident"}
    )

    assert options["allow_source_backfill"] is False
    assert options["allow_seed_source_backfill"] is True
    assert options["allow_completing_seed_source_backfill"] is True
    assert options["teacher_edge_order"] == "dynamic-seed-confidence"
    assert options["teacher_action_filter"] == "seed-source-backfill"
    assert options["teacher_feature_preset"] == "seed-source-cell-confident"
    assert options["min_component_observations"] == 2
    assert options["max_applied_edits"] == 3


def test_moderate_iou_repair_preset_expands_to_seed_source_gate():
    options = _expand_teacher_repair_preset(
        {"teacher_repair_preset": "missing-seed-moderate-iou"}
    )

    assert options["allow_source_backfill"] is False
    assert options["allow_seed_source_backfill"] is True
    assert options["allow_completing_seed_source_backfill"] is True
    assert options["teacher_edge_order"] == "dynamic-seed-confidence"
    assert options["teacher_action_filter"] == "seed-source-backfill"
    assert options["teacher_feature_preset"] == "seed-source-moderate-iou"
    assert options["min_component_observations"] == 2
    assert options["max_applied_edits"] == 2


def test_track2p_fn_repair_preset_expands_to_target_extension_gate():
    options = _expand_teacher_repair_preset(
        {"teacher_repair_preset": "track2p-fn-high-confidence"}
    )

    assert options["teacher_action_filter"] == "target-extension"
    assert options["teacher_edge_order"] == "dynamic-confidence"
    assert options["teacher_feature_preset"] == "track2p-fn-rescue"
    assert options["min_component_observations"] == 2
    assert options["max_applied_edits"] == 3


def test_track2p_fn_moderate_iou_preset_expands_to_target_extension_gate():
    options = _expand_teacher_repair_preset(
        {"teacher_repair_preset": "track2p-fn-moderate-iou-cell-confident"}
    )

    assert options["teacher_action_filter"] == "target-extension"
    assert options["teacher_edge_order"] == "dynamic-confidence"
    assert options["teacher_feature_preset"] == "moderate-iou-cell-confidence"
    assert options["min_component_observations"] == 2
    assert options["max_applied_edits"] == 3


def test_track2p_fn_moderate_iou_confidence_alias_expands_to_target_extension_gate():
    options = _expand_teacher_repair_preset(
        {"teacher_repair_preset": "track2p-fn-moderate-iou-cell-confidence"}
    )

    assert options["teacher_action_filter"] == "target-extension"
    assert options["teacher_edge_order"] == "dynamic-confidence"
    assert options["teacher_feature_preset"] == "moderate-iou-cell-confidence"
    assert options["min_component_observations"] == 2
    assert options["max_applied_edits"] == 3


def test_residual_union_preset_expands_to_two_residual_buckets():
    options = _expand_teacher_repair_preset(
        {"teacher_repair_preset": "residual-union-cell-confident"}
    )

    assert options["allow_source_backfill"] is False
    assert options["allow_seed_source_backfill"] is True
    assert options["allow_completing_seed_source_backfill"] is True
    assert options["allow_fragment_merges"] is False
    assert (
        options["teacher_action_filter"] == "target-extension-or-seed-source-backfill"
    )
    assert options["teacher_edge_order"] == "dynamic-seed-confidence"
    assert options["teacher_feature_preset"] == "residual-fn-cell-confident"
    assert options["min_component_observations"] == 2
    assert options["max_applied_edits"] == 3


def test_missing_seed_repair_preset_preserves_explicit_overrides():
    options = _expand_teacher_repair_preset(
        {
            "teacher_repair_preset": "missing-seed-high-confidence",
            "teacher_edge_order": "dynamic-confidence",
            "teacher_action_filter": "all",
            "teacher_feature_preset": "cell-high-confidence",
            "allow_source_backfill": True,
            "max_applied_edits": 1,
            "min_component_observations": 5,
        }
    )

    assert options["teacher_edge_order"] == "dynamic-confidence"
    assert options["teacher_action_filter"] == "all"
    assert options["teacher_feature_preset"] == "cell-high-confidence"
    assert options["allow_source_backfill"] is True
    assert options["max_applied_edits"] == 1
    assert options["min_component_observations"] == 5


def test_none_repair_preset_is_noop():
    original = {"teacher_repair_preset": "none", "max_applied_edits": 1}

    assert _expand_teacher_repair_preset(original) == original
