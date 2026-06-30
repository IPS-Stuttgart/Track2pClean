from __future__ import annotations

import json

import pytest

from bayescatrack.experiments import benchmark_manifest as bm
from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _teacher_rescue_manifest(min_side_observations):
    return {
        "defaults": {"data": "data", "input_format": "suite2p"},
        "runs": [
            {
                "name": "teacher-rescue-min-side-validation",
                "runner": "track2p-policy-teacher-adjacent-rescue",
                "min_side_observations": min_side_observations,
                "output": "results/teacher-rescue-min-side-validation.csv",
            }
        ],
    }


def test_teacher_rescue_manifest_rejects_boolean_min_side_observations(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(manifest_path, _teacher_rescue_manifest(True))
    run = load_benchmark_manifest(manifest_path).runs[0]

    from bayescatrack.experiments import (  # pylint: disable=import-outside-toplevel
        track2p_policy_teacher_adjacent_rescue as rescue_module,
    )

    def _fake_runner(_config, **_kwargs):
        raise AssertionError("invalid min_side_observations reached runner")

    monkeypatch.setattr(
        rescue_module,
        "run_track2p_policy_teacher_adjacent_rescue",
        _fake_runner,
    )

    with pytest.raises(ValueError, match="min_side_observations"):
        bm._run_benchmark_rows(run)


def test_teacher_rescue_manifest_accepts_string_min_side_observations(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(manifest_path, _teacher_rescue_manifest("3"))
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
    cleanup_config = captured["cleanup_config"]
    assert cleanup_config.min_side_observations == 3
