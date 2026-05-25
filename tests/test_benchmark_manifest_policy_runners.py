from pathlib import Path

from bayescatrack.experiments.benchmark_manifest import (
    _run_config,
    _run_track2p_policy_rows,
    _runner_kwargs,
    _runner_name,
)
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig


def test_policy_runner_aliases_are_supported() -> None:
    assert _runner_name("track2p-policy") == "track2p-policy"
    assert _runner_name("track2p-policy-dp") == "track2p-policy-dp"


def test_policy_runner_config_defaults_method() -> None:
    config = _run_config(
        "track2p-policy",
        {
            "data": "data-root",
            "reference": "gt-root",
            "reference_kind": "manual-gt",
        },
        base_dir=Path("/tmp/benchmark"),
    )

    assert config.method == "global-assignment"
    assert config.data == Path("/tmp/benchmark/data-root")
    assert config.reference == Path("/tmp/benchmark/gt-root")
    assert config.include_non_cells is False
    assert config.weighted_masks is False
    assert config.weighted_centroids is False
    assert config.exclude_overlapping_pixels is False


def test_policy_runner_kwargs_are_separate_from_track2p_config() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "max_gap": 3,
        },
        "track2p-policy",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "max_gap": 3,
    }


def test_policy_manifest_runner_passes_configured_max_gap(monkeypatch) -> None:
    from bayescatrack.experiments import track2p_policy_benchmark

    calls: list[dict[str, object]] = []

    class DummyResult:
        def to_dict(self) -> dict[str, str]:
            return {"variant": "sentinel"}

    def fake_run_track2p_policy_benchmark(
        config: Track2pBenchmarkConfig, **kwargs: object
    ) -> list[DummyResult]:
        calls.append(kwargs)
        return [DummyResult()]

    monkeypatch.setattr(
        track2p_policy_benchmark,
        "run_track2p_policy_benchmark",
        fake_run_track2p_policy_benchmark,
    )

    config = Track2pBenchmarkConfig(
        data=Path("data-root"),
        method="global-assignment",
        max_gap=3,
    )

    rows = _run_track2p_policy_rows(config, {"max_gap": 3})

    assert rows == [{"variant": "sentinel"}]
    assert calls
    assert calls[0]["max_gap"] == 3


def test_policy_dp_runner_kwargs_include_only_dp_specific_options() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "row_top_k": 2,
            "rescue_min_iou": 0.1,
            "max_gap": 2,
        },
        "track2p-policy-dp",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "row_top_k": 2,
        "rescue_min_iou": 0.1,
    }
