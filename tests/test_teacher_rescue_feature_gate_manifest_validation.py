from __future__ import annotations

import json

import pytest

from bayescatrack.experiments import benchmark_manifest as bm
from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _teacher_rescue_manifest(**runner_options):
    run = {
        "name": "teacher-rescue-feature-gate-validation",
        "runner": "track2p-policy-teacher-adjacent-rescue",
        "output": "results/teacher-rescue-feature-gate-validation.csv",
    }
    run.update(runner_options)
    return {
        "defaults": {"data": "data", "input_format": "suite2p"},
        "runs": [run],
    }


@pytest.mark.parametrize(
    "option_name",
    [
        "teacher_min_registered_iou",
        "teacher_gate_min_registered_iou",
        "teacher_max_centroid_distance",
    ],
)
@pytest.mark.parametrize("invalid_value", [[], "nan"])
def test_teacher_rescue_manifest_rejects_malformed_feature_gate_float(
    tmp_path, monkeypatch, option_name, invalid_value
):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(manifest_path, _teacher_rescue_manifest(**{option_name: invalid_value}))
    run = load_benchmark_manifest(manifest_path).runs[0]

    from bayescatrack.experiments import (  # pylint: disable=import-outside-toplevel
        track2p_policy_teacher_adjacent_rescue as rescue_module,
    )

    def _fake_runner(_config, **_kwargs):
        raise AssertionError("invalid feature-gate value reached runner")

    monkeypatch.setattr(
        rescue_module,
        "run_track2p_policy_teacher_adjacent_rescue",
        _fake_runner,
    )

    with pytest.raises(ValueError, match=option_name):
        bm._run_benchmark_rows(run)


def test_teacher_rescue_manifest_normalizes_string_feature_gate_alias(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        _teacher_rescue_manifest(teacher_gate_min_registered_iou="0.25"),
    )
    run = load_benchmark_manifest(manifest_path).runs[0]

    from bayescatrack.experiments import (  # pylint: disable=import-outside-toplevel
        track2p_policy_teacher_adjacent_rescue as rescue_module,
    )

    captured: dict[str, object] = {}

    class _FakeResult:
        def to_dict(self):
            return {"subject": "dummy"}

    class _FakeOutput:
        results = (_FakeResult(),)

    def _fake_runner(_config, **kwargs):
        captured.update(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        rescue_module,
        "run_track2p_policy_teacher_adjacent_rescue",
        _fake_runner,
    )

    rows = bm._run_benchmark_rows(run)

    assert rows == [{"subject": "dummy"}]
    teacher_feature_gate = captured["teacher_feature_gate"]
    assert teacher_feature_gate.min_registered_iou == pytest.approx(0.25)
