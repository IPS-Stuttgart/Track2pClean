from __future__ import annotations

import csv
import importlib
import json

import pytest
from bayescatrack.datasets.track2p import (
    SyntheticTrack2pSubjectConfig,
    write_synthetic_track2p_subject,
)
from bayescatrack.experiments import benchmark_manifest as bm
from bayescatrack.experiments.benchmark_manifest import (
    load_benchmark_manifest,
    run_benchmark_manifest,
)
from bayescatrack.experiments.benchmark_manifest_resolver import (
    resolve_benchmark_manifest_placeholders,
)
from tests._support import run_module


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _read_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_benchmark_manifest_runs_suite_and_comparison(tmp_path):
    write_synthetic_track2p_subject(
        tmp_path / "data",
        SyntheticTrack2pSubjectConfig(subject_name="jm_manifest"),
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data/jm_manifest",
                "method": "track2p-baseline",
                "input_format": "suite2p",
                "include_behavior": False,
            },
            "runs": [
                {
                    "name": "track2p-default",
                    "output": "results/track2p.csv",
                },
                {
                    "name": "repeat-default",
                },
            ],
            "comparisons": [
                {
                    "name": "summary",
                    "inputs": {
                        "Track2p": "track2p-default",
                        "Repeat": "repeat-default",
                    },
                    "output": "results/comparison.md",
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert [run.name for run in result.runs] == ["track2p-default", "repeat-default"]
    assert (tmp_path / "results" / "track2p.csv").exists()
    assert (tmp_path / "benchmark-results" / "repeat-default.csv").exists()
    assert (tmp_path / "results" / "comparison.md").exists()
    assert (
        _read_csv_rows(tmp_path / "results" / "track2p.csv")[0]["reference_source"]
        == "ground_truth_csv"
    )
    assert "Track2p" in (tmp_path / "results" / "comparison.md").read_text(
        encoding="utf-8"
    )


def test_benchmark_manifest_rejects_unknown_run_keys(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "track2p-baseline",
            },
            "runs": [
                {
                    "name": "bad",
                    "unexpected": True,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="unexpected"):
        load_benchmark_manifest(manifest_path)


def test_benchmark_manifest_allows_overlapping_track2p_config_keys(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "global-assignment",
            },
            "runs": [
                {
                    "name": "global",
                    "gap_penalty": 0.6,
                }
            ],
        },
    )

    manifest = load_benchmark_manifest(manifest_path)

    assert manifest.runs[0].config.gap_penalty == 0.6


def test_benchmark_manifest_accepts_hgb_loso_runner_options(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "global-assignment",
                "split": "leave-one-subject-out",
                "cost": "calibrated",
            },
            "runs": [
                {
                    "name": "hgb-loso",
                    "runner": "track2p-loso-calibration",
                    "feature_names": ["one_minus_iou", "centroid_distance"],
                    "sample_weight_strategy": "balanced",
                    "calibration_model": "hist-gradient-boosting",
                    "calibration_model_kwargs": {"max_iter": 25},
                    "hard_negative_options": {
                        "negative_to_positive_ratio": 2.0,
                        "candidate_top_k_per_anchor": 5,
                        "include_column_candidates": False,
                        "hardness_feature_names": ["one_minus_iou"],
                    },
                }
            ],
        },
    )

    run = load_benchmark_manifest(manifest_path).runs[0]
    runner_kwargs = dict(run.runner_kwargs or {})
    hard_negative_options = runner_kwargs["hard_negative_options"]

    assert run.runner == "track2p-loso-calibration"
    assert run.config.method == "global-assignment"
    assert run.config.split == "leave-one-subject-out"
    assert run.config.cost == "calibrated"
    assert runner_kwargs["feature_names"] == ("one_minus_iou", "centroid_distance")
    assert runner_kwargs["sample_weight_strategy"] == "balanced"
    assert runner_kwargs["model_kind"] == "hist-gradient-boosting"
    assert runner_kwargs["model_kwargs"] == {"max_iter": 25}
    assert hard_negative_options.negative_to_positive_ratio == 2.0
    assert hard_negative_options.candidate_top_k_per_anchor == 5
    assert not hard_negative_options.include_column_candidates
    assert hard_negative_options.hardness_feature_names == ("one_minus_iou",)


def test_benchmark_manifest_accepts_monotone_loso_runner_options(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "global-assignment",
                "split": "leave-one-subject-out",
                "cost": "calibrated",
            },
            "runs": [
                {
                    "name": "monotone-loso",
                    "runner": "track2p-monotone-loso",
                    "feature_names": "one_minus_iou,centroid_distance",
                    "monotone_options": {
                        "monotone_feature_names": ["one_minus_iou"],
                        "max_iter": 12,
                        "max_negatives_per_positive": 3,
                    },
                }
            ],
        },
    )

    run = load_benchmark_manifest(manifest_path).runs[0]
    runner_kwargs = dict(run.runner_kwargs or {})
    options = runner_kwargs["monotone_options"]

    assert run.runner == "track2p-monotone-loso"
    assert runner_kwargs["feature_names"] == (
        "one_minus_iou",
        "centroid_distance",
    )
    assert options.monotone_feature_names == ("one_minus_iou",)
    assert options.max_iter == 12
    assert options.max_negatives_per_positive == 3


def test_benchmark_manifest_accepts_monotone_loso_runner_kwargs(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "global-assignment",
                "split": "leave-one-subject-out",
                "cost": "calibrated",
            },
            "runs": [
                {
                    "name": "monotone-loso",
                    "runner": "track2p-monotone-loso",
                    "monotone_ranker_kwargs": {
                        "monotone_feature_names": ["one_minus_iou"],
                        "max_iter": 12,
                    },
                }
            ],
        },
    )

    runner_kwargs = dict(load_benchmark_manifest(manifest_path).runs[0].runner_kwargs)

    assert runner_kwargs["monotone_ranker_kwargs"] == {
        "monotone_feature_names": ["one_minus_iou"],
        "max_iter": 12,
    }


def test_benchmark_manifest_accepts_monotone_loso_runner_kwargs_json(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    monotone_json = '{"max_iter": 12}'
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "global-assignment",
                "split": "leave-one-subject-out",
                "cost": "calibrated",
            },
            "runs": [
                {
                    "name": "monotone-loso",
                    "runner": "track2p-monotone-loso",
                    "monotone_ranker_kwargs_json": monotone_json,
                }
            ],
        },
    )

    runner_kwargs = dict(load_benchmark_manifest(manifest_path).runs[0].runner_kwargs)

    assert runner_kwargs["monotone_ranker_kwargs_json"] == monotone_json


@pytest.mark.parametrize(
    ("first_key", "second_key"),
    [
        ("monotone_options", "monotone_ranker_kwargs"),
        ("monotone_options", "monotone_ranker_kwargs_json"),
        ("monotone_ranker_kwargs", "monotone_ranker_kwargs_json"),
    ],
)
def test_benchmark_manifest_rejects_duplicate_monotone_loso_options(
    tmp_path, first_key, second_key
):
    manifest_path = tmp_path / "benchmarks.json"
    values = {
        "monotone_options": {"max_iter": 10},
        "monotone_ranker_kwargs": {"max_iter": 11},
        "monotone_ranker_kwargs_json": '{"max_iter": 12}',
    }
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "global-assignment",
                "split": "leave-one-subject-out",
                "cost": "calibrated",
            },
            "runs": [
                {
                    "name": "monotone-loso",
                    "runner": "track2p-monotone-loso",
                    first_key: values[first_key],
                    second_key: values[second_key],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="Use either"):
        load_benchmark_manifest(manifest_path)


def test_benchmark_manifest_rejects_duplicate_loso_model_kwargs(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "global-assignment",
                "split": "leave-one-subject-out",
                "cost": "calibrated",
            },
            "runs": [
                {
                    "name": "hgb-loso",
                    "runner": "track2p-loso-calibration",
                    "calibration_model_kwargs": {"max_iter": 25},
                    "calibration_model_kwargs_json": '{"max_iter": 50}',
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="calibration_model_kwargs"):
        load_benchmark_manifest(manifest_path)


@pytest.mark.parametrize(
    ("scalar_key", "scalar_value"),
    [
        ("hard_negative_ratio", 2.0),
        ("hard_negative_top_k", 5),
        ("hard_negative_column_candidates", False),
        ("hard_negative_features", "one_minus_iou"),
    ],
)
def test_benchmark_manifest_rejects_duplicate_hard_negative_options(
    tmp_path, scalar_key, scalar_value
):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "global-assignment",
                "split": "leave-one-subject-out",
                "cost": "calibrated",
            },
            "runs": [
                {
                    "name": "hgb-loso",
                    "runner": "track2p-loso-calibration",
                    "hard_negative_options": {
                        "negative_to_positive_ratio": 4.0,
                    },
                    scalar_key: scalar_value,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="hard_negative_options"):
        load_benchmark_manifest(manifest_path)


def test_benchmark_manifest_rejects_runner_options_for_default_track2p(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "method": "track2p-baseline",
            },
            "runs": [
                {
                    "name": "bad",
                    "feature_names": ["one_minus_iou"],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="does not support.*feature_names"):
        load_benchmark_manifest(manifest_path)


def test_benchmark_manifest_accepts_teacher_adjacent_rescue_options(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "input_format": "suite2p",
            },
            "runs": [
                {
                    "name": "teacher-rescue",
                    "runner": "track2p-policy-teacher-adjacent-rescue",
                    "allow_teacher_complete_row_rescue": True,
                    "allow_teacher_supported_completion": True,
                    "allow_teacher_confirmed_completing_rescue": True,
                    "allow_completing_source_backfill": True,
                    "allow_completing_fragment_merge": True,
                    "allow_seed_completing_backfill": True,
                    "allow_seed_completing_rescue": True,
                    "min_component_observations": 3,
                    "max_applied_edits": 2,
                    "max_target_extension_edits": 1,
                    "max_source_backfill_edits": 0,
                    "max_seed_source_backfill_edits": 1,
                    "max_fragment_merge_edits": 0,
                    "max_completing_rescue_edits": 1,
                    "teacher_edge_order": "dynamic-confidence",
                    "teacher_feature_preset": "high-confidence",
                    "teacher_gate_min_registered_iou": 0.25,
                    "teacher_gate_min_threshold_margin": 0.05,
                    "teacher_gate_min_row_margin": 0.01,
                    "teacher_gate_min_column_margin": 0.02,
                    "teacher_gate_max_centroid_distance": 4.0,
                    "teacher_gate_min_area_ratio": 0.7,
                    "teacher_gate_min_cell_probability": 0.8,
                    "teacher_gate_require_hungarian": True,
                }
            ],
        },
    )

    run = load_benchmark_manifest(manifest_path).runs[0]
    kwargs = dict(run.runner_kwargs or {})

    assert run.runner == "track2p-policy-teacher-adjacent-rescue"
    assert kwargs["allow_teacher_complete_row_rescue"] is True
    assert kwargs["allow_teacher_supported_completion"] is True
    assert kwargs["allow_teacher_confirmed_completing_rescue"] is True
    assert kwargs["allow_completing_source_backfill"] is True
    assert kwargs["allow_completing_fragment_merge"] is True
    assert kwargs["allow_seed_completing_backfill"] is True
    assert kwargs["allow_seed_completing_rescue"] is True
    assert kwargs["min_component_observations"] == 3
    assert kwargs["max_applied_edits"] == 2
    assert kwargs["teacher_edge_order"] == "dynamic-confidence"
    assert kwargs["teacher_feature_preset"] == "high-confidence"
    assert kwargs["teacher_gate_require_hungarian"] is True


def test_benchmark_manifest_dispatches_teacher_adjacent_rescue_options(
    tmp_path, monkeypatch
):
    from bayescatrack.experiments import track2p_policy_teacher_adjacent_rescue

    calls = {}

    class _FakeResult:
        def to_dict(self):
            return {
                "subject": "jm_teacher",
                "variant": "fake teacher rescue",
                "method": "track2p-policy-teacher-adjacent-rescue",
                "n_sessions": 2,
                "reference_source": "ground_truth_csv",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            }

    class _FakeOutput:
        results = (_FakeResult(),)
        component_rows = ()

    def fake_teacher_adjacent_rescue(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = dict(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        track2p_policy_teacher_adjacent_rescue,
        "run_track2p_policy_teacher_adjacent_rescue",
        fake_teacher_adjacent_rescue,
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "teacher-rescue",
                    "runner": "track2p-policy-teacher-adjacent-rescue",
                    "data": "data",
                    "output": "results/teacher-rescue.csv",
                    "allow_completing_source_backfill": True,
                    "allow_seed_completing_rescue": True,
                    "teacher_edge_order": "dynamic-confidence",
                    "teacher_feature_preset": "high-confidence",
                    "teacher_gate_min_registered_iou": 0.25,
                    "teacher_gate_min_cell_probability": 0.8,
                    "teacher_gate_require_hungarian": True,
                    "min_component_observations": 3,
                    "max_applied_edits": 2,
                    "max_target_extension_edits": 1,
                    "max_seed_source_backfill_edits": 1,
                    "max_completing_rescue_edits": 1,
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    gate = calls["kwargs"]["teacher_feature_gate"]
    assert result.runs[0].rows == 1
    assert calls["kwargs"]["allow_completing_source_backfill"] is True
    assert calls["kwargs"]["allow_seed_completing_rescue"] is True
    assert calls["kwargs"]["teacher_edge_order"] == "dynamic-confidence"
    assert calls["kwargs"]["teacher_feature_preset"] == "high-confidence"
    assert calls["kwargs"]["min_component_observations"] == 3
    assert calls["kwargs"]["max_applied_edits"] == 2
    assert calls["kwargs"]["max_target_extension_edits"] == 1
    assert calls["kwargs"]["max_seed_source_backfill_edits"] == 1
    assert calls["kwargs"]["max_completing_rescue_edits"] == 1
    assert gate.min_registered_iou == 0.25
    assert gate.min_cell_probability == 0.8
    assert gate.require_hungarian is True
    assert (tmp_path / "results" / "teacher-rescue.csv").exists()


def test_benchmark_manifest_dispatches_coherence_suffix_teacher_rescue_options(
    tmp_path, monkeypatch
):
    from bayescatrack.experiments import (
        track2p_policy_coherence_suffix_teacher_rescue,
    )

    calls = {}

    class _FakeOutput:
        result_rows = (
            {
                "subject": "jm_teacher_suffix",
                "variant": "fake suffix teacher rescue",
                "method": "track2p-policy-coherence-suffix-teacher-rescue",
                "n_sessions": 2,
                "reference_source": "ground_truth_csv",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            },
        )
        teacher_rows = ()

    def fake_suffix_teacher_rescue(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = dict(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        track2p_policy_coherence_suffix_teacher_rescue,
        "run_track2p_policy_coherence_suffix_teacher_rescue",
        fake_suffix_teacher_rescue,
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "suffix-teacher-rescue",
                    "runner": "track2p-policy-coherence-suffix-teacher-rescue",
                    "data": "data",
                    "output": "results/suffix-teacher-rescue.csv",
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
                    "target_extension_feature_preset": "moderate-iou-cell-confidence",
                    "seed_source_feature_preset": "seed-source-cell-confident",
                    "allow_source_backfill": False,
                    "allow_seed_source_backfill": True,
                    "allow_completing_seed_source_backfill": True,
                    "allow_fragment_merges": False,
                    "min_teacher_component_observations": 2,
                    "max_applied_teacher_edits": -1,
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert result.runs[0].rows == 1
    assert calls["kwargs"]["threshold_method"] == "min"
    assert calls["kwargs"]["iou_distance_threshold"] == 12.0
    assert calls["kwargs"]["cleanup_config"].split_risk_threshold == 1.5
    assert calls["kwargs"]["cleanup_config"].min_side_observations == 2
    assert calls["kwargs"]["suffix_gate"].suffix_path_length == 2
    assert calls["kwargs"]["suffix_gate"].min_cell_probability == 0.8
    assert calls["kwargs"]["suffix_gate"].min_shifted_iou == 0.3
    assert calls["kwargs"]["teacher_edge_order"] == "structural"
    assert calls["kwargs"]["teacher_action_filter"] == "all"
    assert calls["kwargs"]["teacher_feature_preset"] == "none"
    assert (
        calls["kwargs"]["target_extension_feature_preset"]
        == "moderate-iou-cell-confidence"
    )
    assert calls["kwargs"]["seed_source_feature_preset"] == "seed-source-cell-confident"
    assert calls["kwargs"]["allow_source_backfill"] is False
    assert calls["kwargs"]["allow_seed_source_backfill"] is True
    assert calls["kwargs"]["allow_completing_seed_source_backfill"] is True
    assert calls["kwargs"]["allow_fragment_merges"] is False
    assert calls["kwargs"]["min_teacher_component_observations"] == 2
    assert calls["kwargs"]["max_applied_teacher_edits"] is None
    assert (tmp_path / "results" / "suffix-teacher-rescue.csv").exists()


def test_benchmark_manifest_dispatches_growth_veto_cleanup_options(
    tmp_path, monkeypatch
):
    from bayescatrack.experiments import track2p_policy_growth_veto_cleanup

    calls = {}

    class _FakeResult:
        def to_dict(self):
            return {
                "subject": "jm_growth_veto",
                "variant": "fake growth veto cleanup",
                "method": "track2p-policy-growth-veto-cleanup",
                "n_sessions": 7,
                "reference_source": "ground_truth_csv",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            }

    class _FakeOutput:
        results = (_FakeResult(),)
        edge_rows = ()
        summary_rows = ()

    def fake_growth_veto_cleanup(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = dict(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        track2p_policy_growth_veto_cleanup,
        "run_track2p_policy_growth_veto_cleanup",
        fake_growth_veto_cleanup,
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "growth-veto-cleanup",
                    "runner": "track2p-policy-growth-veto-cleanup",
                    "data": "data",
                    "output": "results/growth-veto-cleanup.csv",
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
                    "anchor_min_registered_iou": 0.5,
                    "anchor_min_shifted_iou": 0.3,
                    "anchor_min_cell_probability": 0.8,
                    "min_growth_residual_mahalanobis": 20.0,
                    "min_growth_residual": 3.25,
                    "min_veto_registered_iou": 0.45,
                    "max_veto_registered_iou": 0.6,
                    "min_veto_shifted_iou": 0.6,
                    "max_veto_shifted_iou": 0.8,
                    "max_veto_min_cell_probability": 0.65,
                    "max_veto_local_neighbor_distortion": None,
                    "max_vetoes_per_subject": 1,
                    "growth_veto_base": "coherence-suffix",
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert result.runs[0].rows == 1
    assert calls["kwargs"]["cleanup_config"].split_risk_threshold == 1.5
    assert calls["kwargs"]["suffix_gate"].suffix_path_length == 2
    assert calls["kwargs"]["anchor_min_registered_iou"] == 0.5
    assert calls["kwargs"]["anchor_min_shifted_iou"] == 0.3
    assert calls["kwargs"]["anchor_min_cell_probability"] == 0.8
    assert calls["kwargs"]["prediction_base"] == "coherence-suffix"
    growth_gate = calls["kwargs"]["growth_veto_gate"]
    assert growth_gate.min_growth_residual_mahalanobis == 20.0
    assert growth_gate.min_growth_residual == 3.25
    assert growth_gate.min_registered_iou == 0.45
    assert growth_gate.min_shifted_iou == 0.6
    assert growth_gate.max_min_cell_probability == 0.65
    assert growth_gate.min_anchor_count == 0
    assert growth_gate.min_complete_component_size is None
    assert growth_gate.max_local_neighbor_distortion is None
    assert growth_gate.max_registered_iou == 0.6
    assert growth_gate.max_shifted_iou == 0.8
    assert growth_gate.max_vetoes_per_subject == 1
    assert (tmp_path / "results" / "growth-veto-cleanup.csv").exists()


def test_benchmark_manifest_growth_veto_defaults_keep_upper_iou_caps(
    tmp_path, monkeypatch
):
    from bayescatrack.experiments import track2p_policy_growth_veto_cleanup

    calls = {}

    class _FakeResult:
        def to_dict(self):
            return {
                "subject": "jm_growth_veto_defaults",
                "variant": "fake growth veto cleanup",
                "method": "track2p-policy-growth-veto-cleanup",
                "n_sessions": 7,
                "reference_source": "ground_truth_csv",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            }

    class _FakeOutput:
        results = (_FakeResult(),)
        edge_rows = ()
        summary_rows = ()

    def fake_growth_veto_cleanup(config, **kwargs):
        calls["kwargs"] = dict(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        track2p_policy_growth_veto_cleanup,
        "run_track2p_policy_growth_veto_cleanup",
        fake_growth_veto_cleanup,
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "growth-veto-default-caps",
                    "runner": "track2p-policy-growth-veto-cleanup",
                    "data": "data",
                    "output": "results/growth-veto-default-caps.csv",
                    "min_veto_anchor_count": -3,
                    "min_veto_complete_component_size": None,
                    "max_veto_local_neighbor_distortion": "none",
                }
            ],
        },
    )

    run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    growth_gate = calls["kwargs"]["growth_veto_gate"]
    assert growth_gate.max_registered_iou == 0.60
    assert growth_gate.max_shifted_iou == 0.80
    assert growth_gate.min_anchor_count == 0
    assert growth_gate.min_complete_component_size is None
    assert growth_gate.max_local_neighbor_distortion is None


def test_benchmark_manifest_coherence_suffix_growth_veto_defaults_base(
    tmp_path, monkeypatch
):
    from bayescatrack.experiments import track2p_policy_growth_veto_cleanup

    calls = {}

    class _FakeResult:
        def to_dict(self):
            return {
                "subject": "jm_suffix_growth_veto",
                "variant": "fake coherence suffix growth veto cleanup",
                "method": "track2p-policy-coherence-suffix-growth-veto-cleanup",
                "n_sessions": 7,
                "reference_source": "ground_truth_csv",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            }

    class _FakeOutput:
        results = (_FakeResult(),)
        edge_rows = ()
        summary_rows = ()

    def fake_growth_veto_cleanup(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = dict(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        track2p_policy_growth_veto_cleanup,
        "run_track2p_policy_growth_veto_cleanup",
        fake_growth_veto_cleanup,
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "coherence-suffix-growth-veto",
                    "runner": "track2p-policy-coherence-suffix-growth-veto-cleanup",
                    "data": "data",
                    "output": "results/coherence-suffix-growth-veto.csv",
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert result.runs[0].rows == 1
    assert calls["config"].method == "global-assignment"
    assert calls["kwargs"]["prediction_base"] == "coherence-suffix"
    assert (tmp_path / "results" / "coherence-suffix-growth-veto.csv").exists()


def test_benchmark_manifest_rejects_negative_growth_veto_component_size(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "growth-veto-invalid-component-size",
                    "runner": "track2p-policy-growth-veto-cleanup",
                    "data": "data",
                    "min_veto_complete_component_size": -1,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="min_veto_complete_component_size"):
        run_benchmark_manifest(load_benchmark_manifest(manifest_path))


def test_benchmark_manifest_rejects_ignored_growth_veto_teacher_options(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "growth-veto-ignored-teacher-option",
                    "runner": "track2p-policy-growth-veto-cleanup",
                    "data": "data",
                    "teacher_edge_order": "dynamic-confidence",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="teacher_edge_order"):
        load_benchmark_manifest(manifest_path)


def test_benchmark_manifest_rejects_invalid_growth_veto_base(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "growth-veto-invalid-base",
                    "runner": "track2p-policy-growth-veto-cleanup",
                    "data": "data",
                    "growth_veto_base": "teacher-oracle",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="growth_veto_base"):
        run_benchmark_manifest(load_benchmark_manifest(manifest_path))


@pytest.mark.parametrize(
    ("canonical", "alias"),
    [
        ("min_growth_residual_mahalanobis", "growth_veto_min_mahalanobis"),
        ("min_growth_residual", "growth_veto_min_residual"),
        ("min_veto_registered_iou", "growth_veto_min_registered_iou"),
        ("max_veto_registered_iou", "growth_veto_max_registered_iou"),
        ("min_veto_shifted_iou", "growth_veto_min_shifted_iou"),
        ("max_veto_shifted_iou", "growth_veto_max_shifted_iou"),
        ("min_veto_cell_probability", "growth_veto_min_cell_probability"),
        ("max_veto_min_cell_probability", "growth_veto_max_min_cell_probability"),
        (
            "max_veto_local_neighbor_distortion",
            "growth_veto_max_local_neighbor_distortion",
        ),
        ("min_veto_anchor_count", "growth_veto_min_anchor_count"),
        (
            "min_veto_complete_component_size",
            "growth_veto_min_complete_component_size",
        ),
        ("max_vetoes_per_subject", "growth_veto_max_vetoes_per_subject"),
    ],
)
def test_benchmark_manifest_rejects_duplicate_growth_veto_aliases(
    tmp_path,
    canonical: str,
    alias: str,
) -> None:
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "growth-veto-duplicate-alias",
                    "runner": "track2p-policy-growth-veto-cleanup",
                    "data": "data",
                    canonical: 1,
                    alias: 2,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match=f"{canonical}/{alias}"):
        run_benchmark_manifest(load_benchmark_manifest(manifest_path))


def test_benchmark_manifest_rejects_teacher_base_for_coherence_suffix_growth_veto(
    tmp_path,
):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "suffix-growth-veto-wrong-base",
                    "runner": "track2p-policy-coherence-suffix-growth-veto-cleanup",
                    "data": "data",
                    "growth_veto_base": "teacher-rescue",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="coherence-suffix"):
        run_benchmark_manifest(load_benchmark_manifest(manifest_path))


def test_benchmark_manifest_dispatches_pyrecest_residual_mht_options(
    tmp_path, monkeypatch
):
    track2p_policy_pyrecest_residual_mht_cleanup = importlib.import_module(
        "bayescatrack.experiments.track2p_policy_pyrecest_residual_mht_cleanup"
    )

    calls = {}

    class _FakeResult:
        def to_dict(self):
            return {
                "subject": "jm_pyrecest",
                "variant": "fake PyRecEst residual MHT",
                "method": "track2p-policy-pyrecest-residual-mht-cleanup",
                "n_sessions": 7,
                "reference_source": "ground_truth_csv",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            }

    class _FakeOutput:
        results = (_FakeResult(),)
        candidate_rows = ()
        summary_rows = ()

    def fake_residual_mht(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = dict(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        track2p_policy_pyrecest_residual_mht_cleanup,
        "run_track2p_policy_pyrecest_residual_mht_cleanup",
        fake_residual_mht,
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "pyrecest-dual-pocket",
                    "runner": "track2p-pyrecest-residual-mht-cleanup",
                    "data": "data",
                    "output": "results/pyrecest-dual-pocket.csv",
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
                    "anchor_min_registered_iou": 0.5,
                    "anchor_min_shifted_iou": 0.3,
                    "anchor_min_cell_probability": 0.8,
                    "min_growth_residual_mahalanobis": 12.0,
                    "min_growth_residual": 2.0,
                    "min_veto_registered_iou": 0.35,
                    "max_veto_registered_iou": 0.75,
                    "min_veto_shifted_iou": 0.45,
                    "max_veto_shifted_iou": 0.90,
                    "max_veto_min_cell_probability": 0.65,
                    "max_veto_local_neighbor_distortion": None,
                    "growth_veto_min_anchor_count": -2,
                    "max_veto_row_rank": 2,
                    "max_veto_column_rank": 2,
                    "growth_veto_base": "coherence-suffix",
                    "mht_candidate_top_k": 8,
                    "mht_max_edits_per_subject": 2,
                    "mht_max_hypotheses": 32,
                    "mht_edit_penalty": 0.4,
                    "mht_score_threshold": 1.4,
                    "mht_selection_mode": "global-rescore",
                    "mht_fragmentation_penalty": 0.75,
                    "mht_min_meaningful_track_length": 3,
                    "mht_include_high_overlap_low_motion_candidates": True,
                    "mht_high_overlap_min_registered_iou": 0.9,
                    "mht_high_overlap_max_growth_residual": 1.5,
                    "mht_high_overlap_min_growth_residual_mahalanobis": 1.1,
                    "mht_high_overlap_min_cell_probability": 0.7,
                    "mht_high_overlap_score_bonus": 2.5,
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert result.runs[0].rows == 1
    assert calls["config"].method == "global-assignment"
    assert calls["kwargs"]["threshold_method"] == "min"
    assert calls["kwargs"]["iou_distance_threshold"] == 12.0
    assert calls["kwargs"]["cleanup_config"].split_risk_threshold == 1.5
    assert calls["kwargs"]["suffix_gate"].suffix_path_length == 2
    growth_gate = calls["kwargs"]["growth_veto_gate"]
    assert growth_gate.min_growth_residual_mahalanobis == 12.0
    assert growth_gate.min_growth_residual == 2.0
    assert growth_gate.max_row_rank == 2
    assert growth_gate.max_column_rank == 2
    assert growth_gate.min_anchor_count == 0
    assert growth_gate.max_local_neighbor_distortion is None
    mht_options = calls["kwargs"]["mht_options"]
    assert mht_options.candidate_top_k == 8
    assert mht_options.max_edits_per_subject == 2
    assert mht_options.max_hypotheses == 32
    assert mht_options.edit_penalty == 0.4
    assert mht_options.score_threshold == 1.4
    assert mht_options.selection_mode == "global-rescore"
    assert mht_options.fragmentation_penalty == 0.75
    assert mht_options.min_meaningful_track_length == 3
    assert mht_options.include_high_overlap_low_motion is True
    assert mht_options.high_overlap_min_registered_iou == 0.9
    assert mht_options.high_overlap_max_growth_residual == 1.5
    assert mht_options.high_overlap_min_growth_residual_mahalanobis == 1.1
    assert mht_options.high_overlap_min_cell_probability == 0.7
    assert mht_options.high_overlap_score_bonus == 2.5
    assert (tmp_path / "results" / "pyrecest-dual-pocket.csv").exists()


def test_benchmark_manifest_dispatches_pyrecest_frontier_mht_defaults(
    tmp_path, monkeypatch
):
    track2p_policy_pyrecest_residual_mht_cleanup = importlib.import_module(
        "bayescatrack.experiments.track2p_policy_pyrecest_residual_mht_cleanup"
    )

    calls = []

    class _FakeResult:
        def __init__(self, subject):
            self.subject = subject

        def to_dict(self):
            return {
                "subject": self.subject,
                "variant": "fake PyRecEst frontier MHT",
                "method": "track2p-policy-pyrecest-residual-mht-cleanup",
                "n_sessions": 7,
                "reference_source": "ground_truth_csv",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            }

    class _FakeOutput:
        def __init__(self, subject):
            self.results = (_FakeResult(subject),)
            self.candidate_rows = ()
            self.summary_rows = ()

    def fake_residual_mht(config, **kwargs):
        calls.append((config, dict(kwargs)))
        return _FakeOutput(f"jm_frontier_{len(calls)}")

    monkeypatch.setattr(
        track2p_policy_pyrecest_residual_mht_cleanup,
        "run_track2p_policy_pyrecest_residual_mht_cleanup",
        fake_residual_mht,
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "pyrecest-frontier",
                    "runner": "track2p-pyrecest-frontier-mht-cleanup",
                    "data": "data",
                    "output": "results/pyrecest-frontier.csv",
                    "max_veto_min_cell_probability": 0.70,
                    "mht_score_threshold": 1.70,
                },
                {
                    "name": "pyrecest-safe-frontier",
                    "runner": "track2p-policy-pyrecest-safe-frontier-mht-cleanup",
                    "data": "data",
                    "output": "results/pyrecest-safe-frontier.csv",
                },
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert [run.rows for run in result.runs] == [1, 1]
    assert len(calls) == 2
    first_config, first_kwargs = calls[0]
    second_config, second_kwargs = calls[1]
    assert first_config.method == "global-assignment"
    assert second_config.method == "global-assignment"

    growth_gate = first_kwargs["growth_veto_gate"]
    assert growth_gate.min_growth_residual_mahalanobis == 12.0
    assert growth_gate.min_growth_residual == 2.0
    assert growth_gate.min_registered_iou == 0.35
    assert growth_gate.max_registered_iou == 0.60
    assert growth_gate.min_shifted_iou == 0.45
    assert growth_gate.max_shifted_iou == 0.80
    assert growth_gate.min_cell_probability == 0.50
    assert growth_gate.max_min_cell_probability == 0.70
    assert growth_gate.max_local_neighbor_distortion is None
    assert growth_gate.max_row_rank == 2
    assert growth_gate.max_column_rank == 2
    assert growth_gate.require_not_suffix_edge is True
    assert growth_gate.require_terminal_edge is True
    assert growth_gate.require_last_session_edge is True
    assert growth_gate.require_complete_component is True

    mht_options = first_kwargs["mht_options"]
    assert mht_options.candidate_top_k == 8
    assert mht_options.max_edits_per_subject == 2
    assert mht_options.max_hypotheses == 32
    assert mht_options.edit_penalty == 0.55
    assert mht_options.score_threshold == 1.70

    second_growth_gate = second_kwargs["growth_veto_gate"]
    assert second_growth_gate.max_min_cell_probability == 0.65
    assert second_kwargs["mht_options"].score_threshold == 1.60
    assert (tmp_path / "results" / "pyrecest-frontier.csv").exists()
    assert (tmp_path / "results" / "pyrecest-safe-frontier.csv").exists()


def test_benchmark_manifest_dispatches_pyrecest_calibrated_mht_options(
    tmp_path, monkeypatch
):
    track2p_policy_pyrecest_calibrated_mht_cleanup = importlib.import_module(
        "bayescatrack.experiments.track2p_policy_pyrecest_calibrated_mht_cleanup"
    )

    calls = {}

    class _FakeResult:
        def to_dict(self):
            return {
                "subject": "jm_calibrated",
                "variant": "fake PyRecEst calibrated MHT",
                "method": "track2p-policy-pyrecest-calibrated-mht-cleanup",
                "n_sessions": 7,
                "reference_source": "ground_truth_csv",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            }

    class _FakeOutput:
        results = (_FakeResult(),)
        candidate_rows = ()
        summary_rows = ()

    def fake_calibrated_mht(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = dict(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        track2p_policy_pyrecest_calibrated_mht_cleanup,
        "run_track2p_policy_pyrecest_calibrated_mht_cleanup",
        fake_calibrated_mht,
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "pyrecest-calibrated",
                    "runner": "track2p-policy-pyrecest-calibrated-mht-cleanup",
                    "data": "data",
                    "output": "results/pyrecest-calibrated.csv",
                    "growth_veto_base": "coherence-suffix",
                    "mht_max_edits_per_subject": 3,
                    "mht_max_hypotheses": 48,
                    "mht_edit_penalty": 0.1,
                    "mht_score_threshold": 0.2,
                    "calibrated_fp_logistic_c": 0.75,
                    "calibrated_fp_min_training_positives": 2,
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert result.runs[0].rows == 1
    assert calls["config"].method == "global-assignment"
    mht_options = calls["kwargs"]["mht_options"]
    assert mht_options.max_edits_per_subject == 3
    assert mht_options.max_hypotheses == 48
    assert mht_options.edit_penalty == 0.1
    assert mht_options.score_threshold == 0.2
    assert mht_options.logistic_c == 0.75
    assert mht_options.min_training_positive_examples == 2
    assert calls["kwargs"]["structural_gate"].min_growth_residual == 2.5
    assert calls["kwargs"]["structural_gate"].max_registered_iou == 0.60
    assert calls["kwargs"]["structural_gate"].max_shifted_iou == 0.80
    assert (tmp_path / "results" / "pyrecest-calibrated.csv").exists()


@pytest.mark.parametrize(
    "field",
    [
        "min_growth_residual_mahalanobis",
        "growth_veto_min_mahalanobis",
        "min_growth_residual",
        "growth_veto_min_residual",
        "min_veto_registered_iou",
        "growth_veto_min_registered_iou",
        "max_veto_registered_iou",
        "growth_veto_max_registered_iou",
        "min_veto_shifted_iou",
        "growth_veto_min_shifted_iou",
        "max_veto_shifted_iou",
        "growth_veto_max_shifted_iou",
        "max_vetoes_per_subject",
        "growth_veto_max_vetoes_per_subject",
    ],
)
def test_benchmark_manifest_rejects_ignored_calibrated_mht_growth_veto_fields(
    tmp_path,
    field: str,
) -> None:
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "pyrecest-calibrated",
                    "runner": "track2p-policy-pyrecest-calibrated-mht-cleanup",
                    "data": "data",
                    field: 1,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match=field):
        load_benchmark_manifest(manifest_path)


def test_benchmark_manifest_rejects_invalid_pyrecest_mht_selection_mode(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "pyrecest-invalid-selection",
                    "runner": "track2p-policy-pyrecest-residual-mht-cleanup",
                    "data": "data",
                    "mht_selection_mode": "oracle",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="mht_selection_mode"):
        run_benchmark_manifest(load_benchmark_manifest(manifest_path))


def test_benchmark_manifest_dispatches_configurable_loso_runner(tmp_path, monkeypatch):
    calls = {}

    def fake_configurable_loso(config, options):
        calls["config"] = config
        calls["options"] = dict(options)
        return [
            {
                "subject": "jm_loso",
                "variant": "fake configurable LOSO",
                "method": "global-assignment",
                "n_sessions": 2,
                "reference_source": "ground_truth_csv",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            }
        ]

    monkeypatch.setattr(bm, "_run_configurable_loso_rows", fake_configurable_loso)
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "input_format": "suite2p",
                "include_behavior": False,
            },
            "runs": [
                {
                    "name": "configurable-loso",
                    "runner": "track2p-loso-calibration",
                    "output": "results/configurable-loso.csv",
                    "feature_names": ["registered_iou_cost", "centroid_distance"],
                    "sample_weight_strategy": "balanced",
                    "calibration_model": "hist-gradient-boosting",
                    "calibration_model_kwargs": {"max_iter": 25},
                    "hard_negative_ratio": 2.0,
                    "hard_negative_top_k": 5,
                    "hard_negative_column_candidates": False,
                    "hard_negative_features": "registered_iou_cost",
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert result.runs[0].rows == 1
    assert calls["config"].data == tmp_path / "data"
    assert calls["config"].method == "global-assignment"
    assert calls["config"].split == "leave-one-subject-out"
    assert calls["config"].cost == "calibrated"
    assert calls["config"].include_behavior is False
    assert calls["options"]["sample_weight_strategy"] == "balanced"
    assert calls["options"]["calibration_model_kwargs"] == {"max_iter": 25}
    assert (tmp_path / "results" / "configurable-loso.csv").exists()


def test_benchmark_manifest_dispatches_registration_qa_runner(tmp_path, monkeypatch):
    calls = {}

    def fake_registration_qa(config, options):
        calls["config"] = config
        calls["options"] = dict(options)
        return [
            {
                "cost": "registered-iou",
                "registration_backend": "suite2p-affine",
                "transform_type": "affine",
                "edge_count": 1,
            }
        ]

    monkeypatch.setattr(bm, "_run_registration_qa_rows", fake_registration_qa)
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "backend-audit",
                    "runner": "registration-qa",
                    "data": "data",
                    "level": "backend-audit",
                    "format": "json",
                    "output": "results/backend-audit.json",
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert result.runs[0].rows == 1
    assert calls["config"].data == tmp_path / "data"
    assert calls["options"]["level"] == "backend-audit"
    output_rows = json.loads((tmp_path / "results" / "backend-audit.json").read_text())
    assert output_rows[0]["registration_backend"] == "suite2p-affine"


def test_benchmark_suite_cli_runs_manifest(tmp_path):
    write_synthetic_track2p_subject(
        tmp_path / "data",
        SyntheticTrack2pSubjectConfig(subject_name="jm_cli_manifest"),
    )
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data/jm_cli_manifest",
                "method": "track2p-baseline",
                "input_format": "suite2p",
                "include_behavior": False,
            },
            "runs": [
                {
                    "name": "track2p-default",
                    "output": "results/track2p.csv",
                }
            ],
        },
    )

    proc = run_module(
        "-m",
        "bayescatrack",
        "benchmark",
        "suite",
        str(manifest_path),
        "--summary-format",
        "table",
        "--no-progress",
    )

    assert "track2p-default" in proc.stdout
    assert (tmp_path / "results" / "track2p.csv").exists()


def test_benchmark_manifest_resolver_writes_executable_copy(tmp_path):
    template_path = tmp_path / "template.json"
    output_path = tmp_path / "bundle" / "manifest_resolved.json"
    data_root = tmp_path / "track2p data"
    reference_root = tmp_path / "manual gt"
    output_root = tmp_path / "results"
    _write_manifest(
        template_path,
        {
            "defaults": {
                "data": "<DATA_ROOT>",
                "reference": "<REFERENCE_ROOT>",
                "method": "track2p-baseline",
                "input_format": "suite2p",
            },
            "runs": [
                {
                    "name": "track2p-default",
                    "output": "<OUTPUT_ROOT>/track2p.csv",
                }
            ],
            "comparisons": [
                {
                    "name": "summary",
                    "inputs": {"Track2p": "track2p-default"},
                    "output": "<OUTPUT_ROOT>/comparison.md",
                }
            ],
        },
    )

    resolved_path = resolve_benchmark_manifest_placeholders(
        template_path,
        data_root=data_root,
        reference_root=reference_root,
        output_root=output_root,
        output=output_path,
    )

    assert resolved_path == output_path
    assert "<DATA_ROOT>" in template_path.read_text(encoding="utf-8")
    resolved_text = output_path.read_text(encoding="utf-8")
    assert "<DATA_ROOT>" not in resolved_text
    assert "<REFERENCE_ROOT>" not in resolved_text
    assert "<OUTPUT_ROOT>" not in resolved_text

    manifest = load_benchmark_manifest(output_path)
    assert manifest.runs[0].config.data == data_root
    assert manifest.runs[0].config.reference == reference_root
    assert manifest.runs[0].output == output_root / "track2p.csv"
    assert manifest.comparisons[0].output == output_root / "comparison.md"


def test_benchmark_resolve_suite_cli_writes_manifest(tmp_path):
    template_path = tmp_path / "template.json"
    output_path = tmp_path / "resolved.json"
    _write_manifest(
        template_path,
        {
            "defaults": {
                "data": "<DATA_ROOT>",
                "reference": "<REFERENCE_ROOT>",
                "method": "track2p-baseline",
            },
            "runs": [{"name": "track2p-default", "output": "<OUTPUT_ROOT>/row.csv"}],
        },
    )

    proc = run_module(
        "-m",
        "bayescatrack",
        "benchmark",
        "resolve-suite",
        str(template_path),
        "--data-root",
        str(tmp_path / "data"),
        "--reference-root",
        str(tmp_path / "reference"),
        "--output-root",
        str(tmp_path / "results"),
        "--output",
        str(output_path),
    )

    assert json.loads(proc.stdout)["output"] == str(output_path)
    assert output_path.exists()
    assert "<OUTPUT_ROOT>" not in output_path.read_text(encoding="utf-8")
