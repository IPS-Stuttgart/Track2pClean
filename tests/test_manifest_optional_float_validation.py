from __future__ import annotations

import json

import pytest

from bayescatrack.experiments import benchmark_manifest as bm
from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _teacher_rescue_manifest(**run_overrides):
    run = {
        "name": "teacher-rescue-optional-float-validation",
        "runner": "track2p-policy-teacher-adjacent-rescue",
        "output": "results/teacher-rescue-optional-float-validation.csv",
    }
    run.update(run_overrides)
    return {
        "defaults": {"data": "data", "input_format": "suite2p"},
        "runs": [run],
    }


def test_benchmark_manifest_optional_float_rejects_container_value():
    with pytest.raises(ValueError, match="max_veto_registered_iou"):
        bm._optional_float_option(  # pylint: disable=protected-access
            {"max_veto_registered_iou": []},
            "max_veto_registered_iou",
        )


def test_teacher_rescue_manifest_rejects_container_feature_gate_value(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        _teacher_rescue_manifest(teacher_min_registered_iou=[]),
    )
    run = load_benchmark_manifest(manifest_path).runs[0]

    from bayescatrack.experiments import (  # pylint: disable=import-outside-toplevel
        track2p_policy_teacher_adjacent_rescue as rescue_module,
    )

    def _fake_runner(_config, **_kwargs):
        raise AssertionError("invalid teacher_min_registered_iou reached runner")

    monkeypatch.setattr(
        rescue_module,
        "run_track2p_policy_teacher_adjacent_rescue",
        _fake_runner,
    )

    with pytest.raises(ValueError, match="teacher_min_registered_iou"):
        bm._run_benchmark_rows(run)
