from pathlib import Path

from bayescatrack.experiments.benchmark_manifest import (
    _run_config,
    _runner_kwargs,
    _runner_name,
)


def test_policy_component_runner_aliases_are_supported() -> None:
    assert (
        _runner_name("track2p-policy-component-audit")
        == "track2p-policy-component-audit"
    )
    assert _runner_name("track2p-component-cleanup") == (
        "track2p-policy-component-audit"
    )
    assert (
        _runner_name("track2p-policy-coherence-suffix-stitch")
        == "track2p-policy-coherence-suffix-stitch"
    )
    assert _runner_name("track2p-coherence-suffix-stitch") == (
        "track2p-policy-coherence-suffix-stitch"
    )
    assert _runner_name("track2p-policy-coherence-suffix-teacher-rescue") == (
        "track2p-policy-coherence-suffix-teacher-rescue"
    )
    assert _runner_name("track2p-coherence-suffix-teacher-rescue") == (
        "track2p-policy-coherence-suffix-teacher-rescue"
    )
    assert _runner_name("track2p-policy-coherence-suffix-growth-veto-cleanup") == (
        "track2p-policy-coherence-suffix-growth-veto-cleanup"
    )
    assert _runner_name("track2p-coherence-suffix-growth-veto-cleanup") == (
        "track2p-policy-coherence-suffix-growth-veto-cleanup"
    )
    assert _runner_name("track2p-component-coherence-suffix-growth-veto-cleanup") == (
        "track2p-policy-coherence-suffix-growth-veto-cleanup"
    )


def test_policy_component_runner_config_defaults_method() -> None:
    config = _run_config(
        "track2p-policy-component-audit",
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


def test_policy_coherence_suffix_runner_config_defaults_method() -> None:
    config = _run_config(
        "track2p-policy-coherence-suffix-stitch",
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


def test_policy_coherence_suffix_teacher_runner_config_defaults_method() -> None:
    config = _run_config(
        "track2p-policy-coherence-suffix-teacher-rescue",
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


def test_policy_coherence_suffix_growth_veto_runner_config_defaults_method() -> None:
    config = _run_config(
        "track2p-policy-coherence-suffix-growth-veto-cleanup",
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


def test_policy_component_runner_kwargs_are_runner_specific() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "apply_splits": True,
            "split_risk_threshold": 1.5,
            "min_side_observations": 2,
        },
        "track2p-policy-component-audit",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "apply_splits": True,
        "split_risk_threshold": 1.5,
        "min_side_observations": 2,
    }


def test_policy_coherence_suffix_runner_kwargs_are_runner_specific() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "apply_splits": True,
            "split_risk_threshold": 1.5,
            "min_side_observations": 2,
            "suffix_path_length": 2,
            "min_cell_probability": 0.8,
            "min_area_ratio": 0.8,
            "max_centroid_distance": 6.0,
            "min_shifted_iou": 0.3,
            "min_motion_consistency": 0.5,
            "min_shape_consistency": 0.82,
            "max_stitches_per_subject": 1,
        },
        "track2p-policy-coherence-suffix-stitch",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "split_risk_threshold": 1.5,
        "min_side_observations": 2,
        "suffix_path_length": 2,
        "min_cell_probability": 0.8,
        "min_area_ratio": 0.8,
        "max_centroid_distance": 6.0,
        "min_shifted_iou": 0.3,
        "min_motion_consistency": 0.5,
        "min_shape_consistency": 0.82,
        "max_stitches_per_subject": 1,
    }


def test_policy_coherence_suffix_teacher_runner_kwargs_are_runner_specific() -> None:
    # jscpd:ignore-start
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "apply_splits": True,
            "split_risk_threshold": 1.5,
            "min_side_observations": 2,
            "suffix_path_length": 2,
            "min_cell_probability": 0.8,
            "min_area_ratio": 0.8,
            "max_centroid_distance": 6.0,
            "min_shifted_iou": 0.3,
            "min_motion_consistency": 0.5,
            "min_shape_consistency": 0.82,
            "max_stitches_per_subject": 1,
            "teacher_edge_order": "structural",
            "teacher_action_filter": "all",
            "teacher_feature_preset": "none",
            "max_applied_teacher_edits": -1,
        },
        "track2p-policy-coherence-suffix-teacher-rescue",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "split_risk_threshold": 1.5,
        "min_side_observations": 2,
        "suffix_path_length": 2,
        "min_cell_probability": 0.8,
        "min_area_ratio": 0.8,
        "max_centroid_distance": 6.0,
        "min_shifted_iou": 0.3,
        "min_motion_consistency": 0.5,
        "min_shape_consistency": 0.82,
        "max_stitches_per_subject": 1,
        "teacher_edge_order": "structural",
        "teacher_action_filter": "all",
        "teacher_feature_preset": "none",
        "max_applied_teacher_edits": -1,
    }
    # jscpd:ignore-end
