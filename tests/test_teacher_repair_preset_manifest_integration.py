from __future__ import annotations

import importlib

from bayescatrack.experiments import _teacher_rescue_manifest_integration as base
from bayescatrack.experiments import (
    _teacher_rescue_repair_preset_manifest_integration as repair,
)
from bayescatrack.experiments import benchmark_manifest as bm
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig


def test_teacher_repair_preset_expands_missing_seed_macro():
    expanded = repair._expand_teacher_repair_preset(
        {
            "teacher_repair_preset": "missing-seed-high-confidence",
            "max_applied_edits": 1,
        }
    )

    assert expanded["allow_seed_source_backfill"] is True
    assert expanded["allow_completing_seed_source_backfill"] is True
    assert expanded["teacher_edge_order"] == "dynamic-seed-confidence"
    assert expanded["teacher_feature_preset"] == "seed-source-high-confidence"
    assert expanded["min_component_observations"] == 2
    assert expanded["max_applied_edits"] == 1


def test_teacher_repair_preset_manifest_runner_forwards_macro(monkeypatch, tmp_path):
    captured = {}

    class _FakeResult:
        def to_dict(self):
            return {"subject": "dummy"}

    class _FakeOutput:
        results = (_FakeResult(),)

    def fake_run(config, **kwargs):
        captured.update(kwargs)
        return _FakeOutput()

    from bayescatrack.experiments import (
        track2p_policy_teacher_adjacent_rescue as rescue_module,
    )

    monkeypatch.setattr(
        rescue_module,
        "run_track2p_policy_teacher_adjacent_rescue",
        fake_run,
    )
    config = Track2pBenchmarkConfig(data=tmp_path, method="global-assignment")

    rows = base._run_track2p_policy_teacher_adjacent_rows(
        config,
        {
            "teacher_repair_preset": "missing-seed-high-confidence",
            "allow_source_backfill": False,
            "max_applied_edits": 1,
        },
    )

    assert rows == [{"subject": "dummy"}]
    assert captured["teacher_edge_order"] == "dynamic-seed-confidence"
    assert captured["teacher_feature_preset"] == "seed-source-high-confidence"
    assert captured["allow_seed_source_backfill"] is True
    assert captured["allow_completing_seed_source_backfill"] is True
    assert captured["allow_source_backfill"] is False
    assert captured["min_component_observations"] == 2
    assert captured["max_applied_edits"] == 1


def test_teacher_repair_preset_reinstalls_after_base_reload(monkeypatch, tmp_path):
    reloaded_base = importlib.reload(base)
    repair.install_teacher_rescue_repair_preset_manifest_integration()
    captured = {}

    class _FakeResult:
        def to_dict(self):
            return {"subject": "dummy"}

    class _FakeOutput:
        results = (_FakeResult(),)

    def fake_run(config, **kwargs):
        captured.update(kwargs)
        return _FakeOutput()

    from bayescatrack.experiments import (
        track2p_policy_teacher_adjacent_rescue as rescue_module,
    )

    monkeypatch.setattr(
        rescue_module,
        "run_track2p_policy_teacher_adjacent_rescue",
        fake_run,
    )
    config = Track2pBenchmarkConfig(data=tmp_path, method="global-assignment")

    rows = reloaded_base._run_track2p_policy_teacher_adjacent_rows(
        config,
        {
            "teacher_repair_preset": "missing-seed-high-confidence",
            "max_applied_edits": 1,
        },
    )

    assert rows == [{"subject": "dummy"}]
    assert captured["teacher_edge_order"] == "dynamic-seed-confidence"
    assert captured["teacher_feature_preset"] == "seed-source-high-confidence"
    assert captured["allow_source_backfill"] is False
    assert captured["allow_seed_source_backfill"] is True
    assert captured["allow_completing_seed_source_backfill"] is True
    assert captured["min_component_observations"] == 2
    assert captured["max_applied_edits"] == 1


def test_teacher_repair_preset_registered_as_manifest_field():
    fields = bm._runner_specific_fields("track2p-policy-teacher-adjacent-rescue")

    assert "teacher_repair_preset" in fields
