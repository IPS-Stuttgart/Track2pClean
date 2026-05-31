from __future__ import annotations

import json

from bayescatrack.experiments import benchmark_manifest as bm
from bayescatrack.experiments.advanced_improvement_workbench import (
    track2p_result_improvement_manifest,
)
from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_manifest_accepts_teacher_adjacent_rescue_runner(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {"data": "data", "input_format": "suite2p"},
            "runs": [
                {
                    "name": "teacher-rescue",
                    "runner": "track2p-policy-teacher-adjacent-rescue",
                    "threshold_method": "min",
                    "iou_distance_threshold": 12.0,
                    "cell_probability_threshold": 0.5,
                    "allow_completing_rescue": True,
                    "allow_source_backfill": True,
                    "allow_seed_source_backfill": True,
                    "allow_fragment_merges": True,
                    "output": "results/teacher-rescue.csv",
                }
            ],
        },
    )

    run = load_benchmark_manifest(manifest_path).runs[0]

    assert run.runner == "track2p-policy-teacher-adjacent-rescue"
    assert run.config.data == tmp_path / "data"
    assert run.config.method == "global-assignment"
    assert run.config.include_non_cells is False
    assert run.config.weighted_masks is False
    assert dict(run.runner_kwargs or {})["allow_completing_rescue"] is True


def test_result_improvement_manifest_includes_teacher_adjacent_rescue():
    manifest = track2p_result_improvement_manifest(
        data_root="data",
        reference_root="reference",
        output_root="results",
    )

    run_names = [run["name"] for run in manifest["runs"]]
    assert "track2p-policy-teacher-adjacent-rescue" in run_names
    assert len(run_names) == len(set(run_names))

    teacher = next(
        run
        for run in manifest["runs"]
        if run["name"] == "track2p-policy-teacher-adjacent-rescue"
    )
    assert teacher["runner"] == "track2p-policy-teacher-adjacent-rescue"
    assert teacher["threshold_method"] == "min"
    assert teacher["iou_distance_threshold"] == 12.0
    assert teacher["cell_probability_threshold"] == 0.5
    assert teacher["allow_completing_rescue"] is False
    assert teacher["allow_source_backfill"] is True
    assert teacher["allow_seed_source_backfill"] is False
    assert teacher["allow_fragment_merges"] is True

    for comparison in manifest["comparisons"]:
        labels = list(comparison["inputs"])
        assert "track2p-policy-teacher-adjacent-rescue" in labels


def test_teacher_rescue_runner_specific_fields_registered():
    fields = bm._runner_specific_fields("track2p-policy-teacher-adjacent-rescue")

    assert "allow_completing_rescue" in fields
    assert "allow_seed_source_backfill" in fields
    assert "allow_fragment_merges" in fields
