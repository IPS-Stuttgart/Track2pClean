from __future__ import annotations

from types import SimpleNamespace

from bayescatrack import cli


def test_guarded_gap_consensus_sweep_is_registered() -> None:
    command = cli._BENCHMARK_COMMANDS["track2p-policy-gap-consensus-guarded-sweep"]

    assert (
        command.module
        == "bayescatrack.experiments.track2p_policy_gap_consensus_guarded_sweep"
    )
    assert "adjacent-only" in command.help


def test_guarded_gap_consensus_aliases_resolve_to_registered_command() -> None:
    expected = "track2p-policy-gap-consensus-guarded-sweep"

    assert cli._BENCHMARK_ALIASES["track2p-gap-consensus-guarded-sweep"] == expected
    assert cli._BENCHMARK_ALIASES["track2p-guarded-gap-consensus-sweep"] == expected
    assert (
        cli._BENCHMARK_ALIASES["track2p-gap-rescue-consensus-guarded-sweep"] == expected
    )
    assert expected in cli._BENCHMARK_COMMANDS


def test_guarded_gap_consensus_cli_dispatches_to_guarded_module(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_import_module(module_name: str) -> SimpleNamespace:
        assert (
            module_name
            == "bayescatrack.experiments.track2p_policy_gap_consensus_guarded_sweep"
        )
        return SimpleNamespace(main=lambda argv: calls.append(tuple(argv)) or 0)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    assert (
        cli.main(
            [
                "benchmark",
                "track2p-gap-consensus-guarded-sweep",
                "--data",
                "unused",
            ]
        )
        == 0
    )
    assert calls == [("--data", "unused")]
