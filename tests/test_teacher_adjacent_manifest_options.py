from __future__ import annotations

from pathlib import Path
from typing import Any

from bayescatrack.experiments import _teacher_rescue_manifest_integration as legacy
from bayescatrack.experiments import benchmark_manifest
from bayescatrack.experiments import track2p_policy_teacher_adjacent_rescue as rescue
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig


class _EmptyTeacherRescueOutput:
    results: tuple[Any, ...] = ()


def _minimal_config() -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(data=Path("data"), method="global-assignment")


def test_native_manifest_forwards_action_specific_teacher_feature_presets(
    monkeypatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run_track2p_policy_teacher_adjacent_rescue(
        config: Track2pBenchmarkConfig,
        **kwargs: Any,
    ) -> _EmptyTeacherRescueOutput:
        del config
        seen.update(kwargs)
        return _EmptyTeacherRescueOutput()

    monkeypatch.setattr(
        rescue,
        "run_track2p_policy_teacher_adjacent_rescue",
        fake_run_track2p_policy_teacher_adjacent_rescue,
    )

    benchmark_manifest._run_track2p_policy_teacher_adjacent_rescue_rows(
        _minimal_config(),
        {
            "teacher_repair_preset": "residual-union-action-specific",
            "teacher_feature_preset": "none",
            "target_extension_feature_preset": "moderate-iou-cell-confidence",
            "seed_source_feature_preset": "seed-source-cell-confident",
        },
    )

    assert seen["teacher_repair_preset"] == "residual-union-action-specific"
    assert seen["teacher_feature_preset"] == "none"
    assert seen["target_extension_feature_preset"] == "moderate-iou-cell-confidence"
    assert seen["seed_source_feature_preset"] == "seed-source-cell-confident"


def test_legacy_manifest_integration_forwards_action_specific_teacher_feature_presets(
    monkeypatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run_track2p_policy_teacher_adjacent_rescue(
        config: Track2pBenchmarkConfig,
        **kwargs: Any,
    ) -> _EmptyTeacherRescueOutput:
        del config
        seen.update(kwargs)
        return _EmptyTeacherRescueOutput()

    monkeypatch.setattr(
        rescue,
        "run_track2p_policy_teacher_adjacent_rescue",
        fake_run_track2p_policy_teacher_adjacent_rescue,
    )

    legacy._run_track2p_policy_teacher_adjacent_rows(
        _minimal_config(),
        {
            "teacher_repair_preset": "residual-union-action-specific",
            "teacher_feature_preset": "none",
            "target_extension_feature_preset": "moderate-iou-cell-confidence",
            "seed_source_feature_preset": "seed-source-cell-confident",
        },
    )

    assert seen["teacher_repair_preset"] == "residual-union-action-specific"
    assert seen["teacher_feature_preset"] == "none"
    assert seen["target_extension_feature_preset"] == "moderate-iou-cell-confidence"
    assert seen["seed_source_feature_preset"] == "seed-source-cell-confident"
