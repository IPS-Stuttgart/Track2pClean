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
                    "allow_teacher_supported_completing_rescue": True,
                    "allow_completing_fragment_merges": True,
                    "allow_source_backfill": True,
                    "allow_seed_source_backfill": True,
                    "allow_completing_seed_source_backfill": True,
                    "allow_fragment_merges": True,
                    "min_component_observations": 2,
                    "max_applied_edits": 1,
                    "teacher_min_registered_iou": 0.1,
                    "teacher_max_centroid_distance": 6.0,
                    "teacher_min_area_ratio": 0.45,
                    "teacher_edge_order": "dynamic-confidence",
                    "teacher_feature_preset": "cell-high-confidence",
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
    assert dict(run.runner_kwargs or {})["allow_completing_fragment_merges"] is True
    assert (
        dict(run.runner_kwargs or {})["allow_completing_seed_source_backfill"] is True
    )
    assert (
        dict(run.runner_kwargs or {})["allow_teacher_supported_completing_rescue"]
        is True
    )
    assert dict(run.runner_kwargs or {})["min_component_observations"] == 2
    assert dict(run.runner_kwargs or {})["max_applied_edits"] == 1
    assert dict(run.runner_kwargs or {})["teacher_min_registered_iou"] == 0.1
    assert dict(run.runner_kwargs or {})["teacher_max_centroid_distance"] == 6.0
    assert dict(run.runner_kwargs or {})["teacher_min_area_ratio"] == 0.45
    assert "teacher_min_cell_probability" not in dict(run.runner_kwargs or {})
    assert (
        dict(run.runner_kwargs or {})["teacher_feature_preset"]
        == "cell-high-confidence"
    )
    assert dict(run.runner_kwargs or {})["teacher_edge_order"] == "dynamic-confidence"


def test_teacher_rescue_manifest_passes_teacher_edge_order_to_runner(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {"data": "data", "input_format": "suite2p"},
            "runs": [
                {
                    "name": "teacher-rescue-confidence",
                    "runner": "track2p-policy-teacher-adjacent-rescue",
                    "threshold_method": "min",
                    "iou_distance_threshold": 12.0,
                    "cell_probability_threshold": 0.5,
                    "teacher_edge_order": "confidence",
                    "max_applied_edits": 1,
                    "output": "results/teacher-rescue-confidence.csv",
                }
            ],
        },
    )
    run = load_benchmark_manifest(manifest_path).runs[0]

    from bayescatrack.experiments import (
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
    assert captured["teacher_edge_order"] == "confidence"
    assert captured["max_applied_edits"] == 1
    assert captured["teacher_feature_preset"] == "none"


def test_result_improvement_manifest_includes_teacher_adjacent_rescue_variants():
    manifest = track2p_result_improvement_manifest(
        data_root="data",
        reference_root="reference",
        output_root="results",
    )

    expected = {
        "track2p-policy-teacher-adjacent-rescue": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
        },
        "track2p-policy-teacher-adjacent-rescue-dynamic-structural": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
            "teacher_edge_order": "dynamic-structural",
        },
        "track2p-policy-teacher-adjacent-rescue-confidence": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
            "teacher_edge_order": "confidence",
        },
        "track2p-policy-teacher-adjacent-rescue-feature-gated-dynamic-confidence": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
            "teacher_edge_order": "dynamic-confidence",
            "teacher_min_registered_iou": 0.10,
            "teacher_max_centroid_distance": 6.0,
            "teacher_min_area_ratio": 0.45,
            "min_component_observations": 2,
        },
        "track2p-policy-teacher-adjacent-rescue-feature-gated-dynamic-confidence-max1": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
            "teacher_edge_order": "dynamic-confidence",
            "teacher_min_registered_iou": 0.10,
            "teacher_max_centroid_distance": 6.0,
            "teacher_min_area_ratio": 0.45,
            "min_component_observations": 2,
            "max_applied_edits": 1,
        },
        "track2p-policy-teacher-adjacent-rescue-high-confidence-dynamic-confidence-max1": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
            "teacher_edge_order": "dynamic-confidence",
            "teacher_feature_preset": "high-confidence",
            "max_applied_edits": 1,
        },
        "track2p-policy-teacher-adjacent-rescue-high-confidence-dynamic-confidence-seed-source-max1": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-confidence",
            "teacher_feature_preset": "high-confidence",
            "max_applied_edits": 1,
        },
        "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-max1": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
            "teacher_edge_order": "dynamic-confidence",
            "max_applied_edits": 1,
        },
        "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-max2": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
            "teacher_edge_order": "dynamic-confidence",
            "max_applied_edits": 2,
        },
        "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-seed-source": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-confidence",
        },
        "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-first-edit-seed-source": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-confidence",
            "max_applied_edits": 1,
        },
        "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-seed-source-cellgate": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-confidence",
            "teacher_min_cell_probability": 0.60,
        },
        "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-seed-source-cell-high-confidence-max2": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-confidence",
            "teacher_feature_preset": "cell-high-confidence",
            "max_applied_edits": 2,
        },
        "track2p-policy-teacher-adjacent-rescue-dynamic-seed-confidence-seed-source-max2": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-seed-confidence",
            "teacher_feature_preset": "high-confidence",
            "teacher_min_cell_probability": 0.60,
            "max_applied_edits": 2,
        },
        "track2p-policy-teacher-adjacent-rescue-seed-source": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": False,
        },
        "track2p-policy-teacher-adjacent-rescue-supported": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
            "min_component_observations": 2,
        },
        "track2p-policy-teacher-adjacent-rescue-teacher-completing": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": True,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
        },
        "track2p-policy-teacher-adjacent-rescue-teacher-completing-seed-source": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": True,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
        },
        "track2p-policy-teacher-adjacent-rescue-completing": {
            "allow_completing_rescue": True,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": False,
            "allow_completing_seed_source_backfill": False,
        },
        "track2p-policy-teacher-adjacent-rescue-completing-seed-source": {
            "allow_completing_rescue": False,
            "allow_teacher_supported_completing_rescue": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
        },
    }
    run_names = [run["name"] for run in manifest["runs"]]
    assert set(expected).issubset(run_names)
    assert len(run_names) == len(set(run_names))

    component_index = run_names.index("track2p-policy-component-cleanup")
    assert run_names[component_index + 1 : component_index + 1 + len(expected)] == list(
        expected
    )
    runs_by_name = {run["name"]: run for run in manifest["runs"]}
    for name, flags in expected.items():
        teacher = runs_by_name[name]
        assert teacher["runner"] == "track2p-policy-teacher-adjacent-rescue"
        assert teacher["threshold_method"] == "min"
        assert teacher["iou_distance_threshold"] == 12.0
        assert teacher["cell_probability_threshold"] == 0.5
        assert teacher["allow_source_backfill"] is True
        assert teacher["allow_fragment_merges"] is True
        assert "min_component_observations" in teacher
        for flag, value in flags.items():
            if isinstance(value, bool):
                assert teacher[flag] is value
            else:
                assert teacher[flag] == value

    for comparison in manifest["comparisons"]:
        labels = list(comparison["inputs"])
        for name in expected:
            assert name in labels
        component_position = labels.index("track2p-policy-component-cleanup")
        expected_labels = list(expected)
        actual_labels = labels[
            component_position + 1 : component_position + 1 + len(expected)
        ]
        assert actual_labels == expected_labels


def test_teacher_rescue_runner_specific_fields_registered():
    fields = bm._runner_specific_fields("track2p-policy-teacher-adjacent-rescue")

    assert "allow_completing_rescue" in fields
    assert "allow_teacher_supported_completing_rescue" in fields
    assert "allow_completing_fragment_merges" in fields
    assert "allow_seed_source_backfill" in fields
    assert "allow_completing_seed_source_backfill" in fields
    assert "allow_fragment_merges" in fields
    assert "min_component_observations" in fields
    assert "max_applied_edits" in fields
    assert "teacher_edge_order" in fields
    assert "teacher_feature_preset" in fields
    assert "teacher_min_registered_iou" in fields
    assert "teacher_min_cell_probability" in fields
    assert "teacher_require_hungarian" in fields


def test_teacher_rescue_manifest_runner_passes_teacher_edge_order(
    monkeypatch, tmp_path
):
    from bayescatrack.experiments import (
        _teacher_rescue_manifest_integration as integration,
    )
    from bayescatrack.experiments import (
        track2p_policy_teacher_adjacent_rescue as rescue_module,
    )
    from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig

    captured = {}

    class _FakeResult:
        def to_dict(self):
            return {"subject": "dummy"}

    class _FakeOutput:
        results = (_FakeResult(),)

    def fake_run(config, **kwargs):
        captured.update(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        rescue_module, "run_track2p_policy_teacher_adjacent_rescue", fake_run
    )
    config = Track2pBenchmarkConfig(data=tmp_path, method="global-assignment")

    rows = integration._run_track2p_policy_teacher_adjacent_rows(
        config, {"teacher_edge_order": "confidence"}
    )

    assert rows == [{"subject": "dummy"}]
    assert captured["teacher_edge_order"] == "confidence"


def test_teacher_rescue_manifest_runner_passes_feature_gate(monkeypatch, tmp_path):
    from bayescatrack.experiments import (
        _teacher_rescue_manifest_integration as integration,
    )
    from bayescatrack.experiments import (
        track2p_policy_teacher_adjacent_rescue as rescue_module,
    )
    from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig

    captured = {}

    class _FakeResult:
        def to_dict(self):
            return {"subject": "dummy"}

    class _FakeOutput:
        results = (_FakeResult(),)

    def fake_run(config, **kwargs):
        captured.update(kwargs)
        return _FakeOutput()

    monkeypatch.setattr(
        rescue_module, "run_track2p_policy_teacher_adjacent_rescue", fake_run
    )
    config = Track2pBenchmarkConfig(data=tmp_path, method="global-assignment")

    rows = integration._run_track2p_policy_teacher_adjacent_rows(
        config,
        {
            "teacher_min_registered_iou": 0.4,
            "teacher_max_centroid_distance": 3.0,
            "teacher_min_cell_probability": 0.7,
            "teacher_require_hungarian": True,
            "max_applied_edits": 1,
            "teacher_feature_preset": "high-confidence",
        },
    )

    assert rows == [{"subject": "dummy"}]
    gate = captured["teacher_feature_gate"]
    assert gate.min_registered_iou == 0.4
    assert gate.max_centroid_distance == 3.0
    assert gate.min_cell_probability == 0.7
    assert gate.require_hungarian is True
    assert captured["max_applied_edits"] == 1
    assert captured["teacher_feature_preset"] == "high-confidence"
