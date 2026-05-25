from __future__ import annotations

import json
import subprocess  # nosec B404

import pytest
from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest
from bayescatrack.experiments.benchmark_manifest_plan import (
    build_manifest_plan,
    format_manifest_plan_table,
    validate_manifest_input_paths,
)
from tests._support import run_module


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_manifest_plan_resolves_runs_and_comparisons_without_running(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "missing-data",
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
                    "name": "registered-iou",
                    "method": "global-assignment",
                    "cost": "registered-iou",
                    "max_gap": 2,
                },
            ],
            "comparisons": [
                {
                    "name": "summary",
                    "inputs": {
                        "Track2p": "track2p-default",
                        "Registered IoU": "registered-iou",
                    },
                    "output": "results/comparison.md",
                }
            ],
        },
    )

    plan = build_manifest_plan(load_benchmark_manifest(manifest_path))

    assert plan["manifest"] == str(manifest_path)
    assert [row["name"] for row in plan["runs"]] == [
        "track2p-default",
        "registered-iou",
    ]
    assert plan["runs"][0]["data"] == str(tmp_path / "missing-data")
    assert plan["runs"][1]["method"] == "global-assignment"
    assert plan["runs"][1]["cost"] == "registered-iou"
    assert plan["comparisons"][0]["inputs"] == {
        "Track2p": "track2p-default",
        "Registered IoU": "registered-iou",
    }


def test_manifest_plan_table_mentions_outputs_and_runner_options(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "hgb-loso",
                    "runner": "track2p-loso-calibration",
                    "data": "missing-data",
                    "feature_names": ["registered_iou_cost", "centroid_distance"],
                    "calibration_model": "hist-gradient-boosting",
                    "output": "results/hgb.csv",
                }
            ],
        },
    )

    table = format_manifest_plan_table(
        build_manifest_plan(load_benchmark_manifest(manifest_path))
    )

    assert "# Benchmark manifest plan" in table
    assert "hgb-loso" in table
    assert "track2p-loso-calibration" in table
    assert "feature_names" in table
    assert "hgb.csv" in table


def test_validate_manifest_input_paths_is_opt_in(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "track2p-default",
                    "data": "missing-data",
                    "reference": "missing-reference",
                    "method": "track2p-baseline",
                }
            ],
        },
    )
    manifest = load_benchmark_manifest(manifest_path)

    with pytest.raises(FileNotFoundError, match="track2p-default.data") as exc_info:
        validate_manifest_input_paths(manifest)

    assert "track2p-default.reference" in str(exc_info.value)


def test_validate_suite_cli_prints_plan_without_existing_data(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "track2p-default",
                    "data": "missing-data",
                    "method": "track2p-baseline",
                    "output": "results/track2p.csv",
                }
            ],
        },
    )

    proc = run_module(
        "-m",
        "bayescatrack",
        "benchmark",
        "validate-suite",
        str(manifest_path),
        "--format",
        "json",
    )
    payload = json.loads(proc.stdout)

    assert payload["runs"][0]["name"] == "track2p-default"
    assert payload["runs"][0]["data"] == str(tmp_path / "missing-data")


def test_validate_suite_cli_can_check_input_paths(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "track2p-default",
                    "data": "missing-data",
                    "method": "track2p-baseline",
                }
            ],
        },
    )

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_module(
            "-m",
            "bayescatrack",
            "benchmark",
            "validate-suite",
            str(manifest_path),
            "--check-input-paths",
        )

    assert "track2p-default.data" in exc_info.value.stderr
