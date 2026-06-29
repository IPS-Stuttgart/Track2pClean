from __future__ import annotations

from pathlib import Path

import pytest
from bayescatrack import accuracy_presets as module
from bayescatrack.accuracy_presets import AccuracyPreset, run_track2p_accuracy_presets
from bayescatrack.experiments.track2p_benchmark import (
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
)


def _fake_config() -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(
        data=Path("/unused"),
        method="global-assignment",
    )


def _fake_result() -> SubjectBenchmarkResult:
    return SubjectBenchmarkResult(
        subject="jm_synthetic",
        variant="synthetic preset",
        method="global-assignment",
        scores={},
        n_sessions=2,
        reference_source="manual_gt",
    )


def test_run_track2p_accuracy_presets_accepts_single_preset_name(monkeypatch) -> None:
    result = _fake_result()
    preset = AccuracyPreset(
        name="registered-shifted-iou-safe",
        description="synthetic preset",
        config=_fake_config(),
    )

    monkeypatch.setattr(
        module,
        "build_track2p_accuracy_presets",
        lambda *args, **kwargs: (preset,),
    )
    monkeypatch.setattr(module, "_run_accuracy_preset", lambda selected: [result])

    output = run_track2p_accuracy_presets(
        "/data/track2p",
        preset_names="registered-shifted-iou-safe",
    )

    assert output == {"registered-shifted-iou-safe": [result]}


def test_run_track2p_accuracy_presets_validates_unknown_names_before_running(
    monkeypatch,
) -> None:
    calls: list[str] = []
    preset = AccuracyPreset(
        name="registered-shifted-iou-safe",
        description="synthetic preset",
        config=_fake_config(),
    )

    def fake_run(selected: AccuracyPreset) -> list[SubjectBenchmarkResult]:
        calls.append(str(selected.name))
        return [_fake_result()]

    monkeypatch.setattr(
        module,
        "build_track2p_accuracy_presets",
        lambda *args, **kwargs: (preset,),
    )
    monkeypatch.setattr(module, "_run_accuracy_preset", fake_run)

    with pytest.raises(ValueError, match="Unknown accuracy preset"):
        run_track2p_accuracy_presets(
            "/data/track2p",
            preset_names=("registered-shifted-iou-safe", "missing-preset"),
        )

    assert calls == []


@pytest.mark.parametrize(
    "preset_names",
    [b"registered-shifted-iou-safe", bytearray(b"registered-shifted-iou-safe")],
)
def test_run_track2p_accuracy_presets_rejects_byte_like_preset_names(
    monkeypatch,
    preset_names: object,
) -> None:
    calls: list[str] = []
    preset = AccuracyPreset(
        name="registered-shifted-iou-safe",
        description="synthetic preset",
        config=_fake_config(),
    )

    def fake_run(selected: AccuracyPreset) -> list[SubjectBenchmarkResult]:
        calls.append(str(selected.name))
        return [_fake_result()]

    monkeypatch.setattr(
        module,
        "build_track2p_accuracy_presets",
        lambda *args, **kwargs: (preset,),
    )
    monkeypatch.setattr(module, "_run_accuracy_preset", fake_run)

    with pytest.raises(
        ValueError,
        match="preset_names must be a preset name or an iterable of preset names",
    ):
        run_track2p_accuracy_presets(
            "/data/track2p",
            preset_names=preset_names,  # type: ignore[arg-type]
        )

    assert calls == []
