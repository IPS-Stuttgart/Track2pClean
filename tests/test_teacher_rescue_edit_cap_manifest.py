from __future__ import annotations

import json

from bayescatrack.experiments.advanced_improvement_workbench import (
    track2p_result_improvement_manifest,
)
from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_manifest_accepts_teacher_rescue_max_applied_edits(tmp_path):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {"data": "data", "input_format": "suite2p"},
            "runs": [
                {
                    "name": "teacher-rescue-max2",
                    "runner": "track2p-policy-teacher-adjacent-rescue",
                    "teacher_edge_order": "dynamic-confidence",
                    "max_applied_edits": 2,
                    "output": "results/teacher-rescue-max2.csv",
                }
            ],
        },
    )

    run = load_benchmark_manifest(manifest_path).runs[0]

    assert run.runner == "track2p-policy-teacher-adjacent-rescue"
    assert dict(run.runner_kwargs or {})["teacher_edge_order"] == "dynamic-confidence"
    assert dict(run.runner_kwargs or {})["max_applied_edits"] == 2


def test_result_improvement_manifest_includes_edit_capped_teacher_rows():
    manifest = track2p_result_improvement_manifest(
        data_root="data",
        reference_root="reference",
        output_root="results",
    )

    max1 = "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-max1"
    max2 = "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-max2"
    anchor = "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-seed-source"
    run_names = [run["name"] for run in manifest["runs"]]

    assert max1 in run_names
    assert max2 in run_names
    anchor_index = run_names.index(anchor)
    assert run_names[anchor_index + 1 : anchor_index + 3] == [max1, max2]

    runs_by_name = {run["name"]: run for run in manifest["runs"]}
    assert runs_by_name[max1]["max_applied_edits"] == 1
    assert runs_by_name[max2]["max_applied_edits"] == 2
