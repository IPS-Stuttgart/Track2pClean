from pathlib import Path

from bayescatrack.experiments.benchmark_manifest import (
    _run_config,
    _runner_kwargs,
    _runner_name,
)


def test_policy_teacher_adjacent_rescue_runner_aliases_are_supported() -> None:
    assert _runner_name("track2p-policy-teacher-adjacent-rescue") == (
        "track2p-policy-teacher-adjacent-rescue"
    )
    assert _runner_name("track2p-teacher-adjacent-rescue") == (
        "track2p-policy-teacher-adjacent-rescue"
    )


def test_policy_teacher_adjacent_rescue_runner_config_defaults_method() -> None:
    config = _run_config(
        "track2p-policy-teacher-adjacent-rescue",
        {
            "data": "data-root",
            "reference": "gt-root",
            "reference_kind": "manual-gt",
        },
        base_dir=Path("/tmp/benchmark"),
    )

    assert config.method == "global-assignment"
    assert config.data == Path("/tmp/benchmark/data-root")
    assert config.reference == Path("/tmp/benchmark/gt-root")
    assert config.include_non_cells is False
    assert config.weighted_masks is False
    assert config.weighted_centroids is False
    assert config.exclude_overlapping_pixels is False


def test_policy_teacher_adjacent_rescue_runner_kwargs_are_runner_specific() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "split_risk_threshold": 1.5,
            "min_side_observations": 2,
            "allow_completing_rescue": True,
            "allow_teacher_supported_completing_rescue": True,
            "allow_completing_fragment_merges": True,
            "apply_splits": True,
            "allow_source_backfill": True,
            "allow_source_inserts": False,
            "allow_source_insertions": True,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "allow_fragment_merges": True,
            "allow_teacher_complete_row_rescue": True,
            "allow_teacher_supported_completion": True,
            "allow_teacher_confirmed_completing_rescue": True,
            "allow_completing_source_backfill": True,
            "allow_completing_fragment_merge": True,
            "allow_seed_completing_backfill": True,
            "allow_seed_completing_rescue": True,
            "teacher_feature_preset": "high-confidence",
            "teacher_edge_order": "dynamic-confidence",
            "teacher_action_filter": "target-extension",
            "teacher_repair_preset": "missing-seed-high-confidence",
            "min_component_observations": 2,
            "max_applied_edits": 2,
            "teacher_min_registered_iou": 0.4,
            "teacher_min_cell_probability": 0.8,
            "teacher_require_hungarian": True,
        },
        "track2p-policy-teacher-adjacent-rescue",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "split_risk_threshold": 1.5,
        "min_side_observations": 2,
        "allow_completing_rescue": True,
        "allow_teacher_supported_completing_rescue": True,
        "allow_completing_fragment_merges": True,
        "allow_source_backfill": True,
        "allow_source_inserts": False,
        "allow_source_insertions": True,
        "allow_seed_source_backfill": True,
        "allow_completing_seed_source_backfill": True,
        "allow_fragment_merges": True,
        "allow_teacher_complete_row_rescue": True,
        "allow_teacher_supported_completion": True,
        "allow_teacher_confirmed_completing_rescue": True,
        "allow_completing_source_backfill": True,
        "allow_completing_fragment_merge": True,
        "allow_seed_completing_backfill": True,
        "allow_seed_completing_rescue": True,
        "teacher_feature_preset": "high-confidence",
        "teacher_edge_order": "dynamic-confidence",
        "teacher_action_filter": "target-extension",
        "teacher_repair_preset": "missing-seed-high-confidence",
        "min_component_observations": 2,
        "max_applied_edits": 2,
        "teacher_min_registered_iou": 0.4,
        "teacher_min_cell_probability": 0.8,
        "teacher_require_hungarian": True,
    }
