"""Tests for Track2p benchmark higher-order consistency plumbing."""

from pathlib import Path
from types import SimpleNamespace

from bayescatrack.association.higher_order_consistency import (
    HigherOrderConsistencyConfig,
)
from bayescatrack.experiments import track2p_benchmark


def test_track2p_benchmark_parser_exposes_higher_order_knobs():
    args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--method",
            "global-assignment",
            "--triplet-weight",
            "0.25",
            "--triplet-support-top-k",
            "4",
            "--triplet-support-cost-cap",
            "1.5",
            "--triplet-max-penalty",
            "0.75",
            "--triplet-large-cost",
            "12345",
        ]
    )

    config = track2p_benchmark._config_from_args(args)

    assert config.triplet_weight == 0.25
    assert config.triplet_support_top_k == 4
    assert config.triplet_support_cost_cap == 1.5
    assert config.triplet_max_penalty == 0.75
    assert config.triplet_large_cost == 12345.0


def test_solve_configured_global_assignment_passes_higher_order_config(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_solve_global_assignment_for_sessions(sessions, **kwargs):
        captured["sessions"] = sessions
        captured.update(kwargs)
        return SimpleNamespace(
            result=SimpleNamespace(tracks=[]),
            pairwise_costs={},
            session_sizes=(),
            session_edges=(),
        )

    monkeypatch.setattr(
        track2p_benchmark,
        "solve_global_assignment_for_sessions",
        _fake_solve_global_assignment_for_sessions,
    )
    config = track2p_benchmark.Track2pBenchmarkConfig(
        data=Path("unused"),
        method="global-assignment",
        triplet_weight=0.25,
        triplet_support_top_k=4,
        triplet_support_cost_cap=1.5,
        triplet_max_penalty=0.75,
        triplet_large_cost=12345.0,
    )

    track2p_benchmark.solve_configured_global_assignment([], config)

    higher_order_config = captured["higher_order_consistency_config"]
    assert isinstance(higher_order_config, HigherOrderConsistencyConfig)
    assert higher_order_config.triplet_weight == 0.25
    assert higher_order_config.support_top_k == 4
    assert higher_order_config.support_cost_cap == 1.5
    assert higher_order_config.max_penalty == 0.75
    assert higher_order_config.large_cost == 12345.0


def test_default_track2p_benchmark_disables_higher_order_config():
    config = track2p_benchmark.Track2pBenchmarkConfig(
        data=Path("unused"),
        method="global-assignment",
    )

    assert track2p_benchmark._higher_order_consistency_config(config) is None
