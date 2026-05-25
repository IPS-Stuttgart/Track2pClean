from __future__ import annotations

import csv
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
