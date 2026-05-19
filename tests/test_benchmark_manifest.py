from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from bayescatrack.datasets.track2p import (
    SyntheticTrack2pSubjectConfig,
    write_synthetic_track2p_subject,
)
from bayescatrack.experiments.benchmark_manifest import (
    load_benchmark_manifest,
    run_benchmark_manifest,
)
from tests._support import run_module


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _read_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_track2p_reference(subject):
    track2p_dir = subject.subject_dir / "track2p"
    track2p_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        track2p_dir / f"{subject.plane_name}_suite2p_indices.npy",
        subject.suite2p_indices,
        allow_pickle=True,
    )
    np.save(
        track2p_dir / "track_ops.npy",
        {
            "all_ds_path": [
                str(subject.subject_dir / name) for name in subject.session_names
            ]
        },
        allow_pickle=True,
    )


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


def test_benchmark_manifest_runs_track2p_teacher_audit_wrapper(tmp_path):
    subject = write_synthetic_track2p_subject(
        tmp_path / "data",
        SyntheticTrack2pSubjectConfig(subject_name="jm_teacher_manifest"),
    )
    _write_track2p_reference(subject)
    manifest_path = tmp_path / "teacher-benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data/jm_teacher_manifest",
                "input_format": "suite2p",
                "include_behavior": False,
            },
            "runs": [
                {
                    "name": "teacher-audit",
                    "benchmark": "track2p-teacher-audit",
                    "pair_mode": "consecutive",
                    "format": "csv",
                    "output": "results/teacher-summary.csv",
                    "edges_output": "results/teacher-edges.csv",
                    "focus_output": "results/teacher-focus.csv",
                    "teacher_output": "results/teacher-labels.csv",
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert [run.name for run in result.runs] == ["teacher-audit"]
    assert result.runs[0].rows == 1
    summary_rows = _read_csv_rows(tmp_path / "results" / "teacher-summary.csv")
    assert float(summary_rows[0]["track2p_vs_gt_f1"]) == pytest.approx(1.0)
    assert "bayes_miss_rate_on_gt_track2p_agreement" in summary_rows[0]
    edge_rows = _read_csv_rows(tmp_path / "results" / "teacher-edges.csv")
    assert edge_rows
    assert {"in_ground_truth", "in_track2p", "in_bayes", "category"}.issubset(
        edge_rows[0]
    )
    teacher_rows = _read_csv_rows(tmp_path / "results" / "teacher-labels.csv")
    assert teacher_rows
    assert teacher_rows[0]["teacher_label_source"] == "track2p_output"
    assert (tmp_path / "results" / "teacher-focus.csv").exists()


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
