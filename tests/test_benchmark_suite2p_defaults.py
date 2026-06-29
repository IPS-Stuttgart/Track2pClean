from __future__ import annotations

from pathlib import Path

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig


def test_programmatic_benchmark_config_keeps_suite2p_non_cells_by_default():
    config = Track2pBenchmarkConfig(
        data=Path("/tmp/track2p"),
        method="track2p-baseline",
    )

    assert config.include_non_cells is True


def test_programmatic_benchmark_config_keeps_explicit_hard_filter_override():
    config = Track2pBenchmarkConfig(
        data=Path("/tmp/track2p"),
        method="track2p-baseline",
        include_non_cells=False,
    )

    assert config.include_non_cells is False
