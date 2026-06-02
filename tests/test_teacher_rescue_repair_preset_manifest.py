from __future__ import annotations

from bayescatrack.experiments._teacher_rescue_repair_preset_manifest_integration import (
    _expand_teacher_repair_preset,
)


def test_missing_seed_repair_preset_expands_to_narrow_defaults():
    options = _expand_teacher_repair_preset(
        {"teacher_repair_preset": "missing-seed-high-confidence"}
    )

    assert options["allow_seed_source_backfill"] is True
    assert options["allow_completing_seed_source_backfill"] is True
    assert options["teacher_edge_order"] == "dynamic-seed-confidence"
    assert options["teacher_feature_preset"] == "seed-source-high-confidence"
    assert options["min_component_observations"] == 2
    assert options["max_applied_edits"] == 2


def test_missing_seed_repair_preset_preserves_explicit_overrides():
    options = _expand_teacher_repair_preset(
        {
            "teacher_repair_preset": "missing-seed-high-confidence",
            "teacher_edge_order": "dynamic-confidence",
            "teacher_feature_preset": "cell-high-confidence",
            "max_applied_edits": 1,
            "min_component_observations": 5,
        }
    )

    assert options["teacher_edge_order"] == "dynamic-confidence"
    assert options["teacher_feature_preset"] == "cell-high-confidence"
    assert options["max_applied_edits"] == 1
    assert options["min_component_observations"] == 5


def test_none_repair_preset_is_noop():
    original = {"teacher_repair_preset": "none", "max_applied_edits": 1}

    assert _expand_teacher_repair_preset(original) == original
