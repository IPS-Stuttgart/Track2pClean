"""Tests for Track2p higher-order consistency ablation plumbing."""

from pathlib import Path

import pytest
from bayescatrack.experiments import track2p_benchmark as benchmark


def _minimal_args() -> list[str]:
    return [
        "--data",
        str(Path(".")),
        "--method",
        "global-assignment",
        "--no-progress",
    ]


def test_higher_order_cli_is_disabled_by_default():
    args = benchmark.build_arg_parser().parse_args(_minimal_args())

    config = benchmark._config_from_args(args)

    assert config.higher_order_consistency_config is None


def test_higher_order_cli_builds_valid_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            *_minimal_args(),
            "--higher-order-triplet-weight",
            "0.25",
            "--higher-order-support-top-k",
            "4",
            "--higher-order-support-cost-cap",
            "1.5",
            "--higher-order-max-penalty",
            "0.75",
            "--higher-order-large-cost",
            "12345",
        ]
    )

    config = benchmark._config_from_args(args)

    assert config.higher_order_consistency_config == {
        "triplet_weight": 0.25,
        "support_top_k": 4,
        "support_cost_cap": 1.5,
        "max_penalty": 0.75,
        "large_cost": 12345.0,
    }


def test_invalid_higher_order_cli_is_rejected():
    args = benchmark.build_arg_parser().parse_args(
        [*_minimal_args(), "--higher-order-support-top-k", "0"]
    )

    with pytest.raises(ValueError, match="support_top_k"):
        benchmark._config_from_args(args)


def test_higher_order_config_is_forwarded_to_solver(monkeypatch):
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_solver(sessions, **kwargs):
        captured["sessions"] = sessions
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        benchmark,
        "solve_global_assignment_for_sessions",
        fake_solver,
    )
    config = benchmark.Track2pBenchmarkConfig(
        data=Path("."),
        method="global-assignment",
        higher_order_consistency_config={"triplet_weight": 0.25},
    )
    sessions = [object()]

    result = benchmark.solve_configured_global_assignment(sessions, config)

    assert result is sentinel
    assert captured["sessions"] == sessions
    assert captured["higher_order_consistency_config"] == {"triplet_weight": 0.25}
