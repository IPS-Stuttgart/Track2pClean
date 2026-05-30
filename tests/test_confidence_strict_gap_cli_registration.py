from __future__ import annotations

from types import SimpleNamespace

from bayescatrack import cli

# pylint: disable=protected-access


def test_confidence_ordered_strict_gap_cleanup_is_registered() -> None:
    canonical = "track2p-policy-confidence-ordered-strict-gated-gap-cleanup"

    assert canonical in cli._BENCHMARK_COMMANDS
    command = cli._BENCHMARK_COMMANDS[canonical]
    assert command.module == (
        "bayescatrack.experiments.track2p_policy_confidence_ordered_strict_gap_cleanup"
    )
    assert "confidence-ordered" in command.help
    assert cli._BENCHMARK_ALIASES["track2p-confidence-strict-gap-cleanup"] == canonical
    assert (
        cli._BENCHMARK_ALIASES["track2p-confidence-ordered-strict-gap-cleanup"]
        == canonical
    )
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-confidence-strict-gap-cleanup"]
        == canonical
    )


def test_confidence_ordered_strict_gap_cli_dispatches_to_module(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_import_module(module_name: str) -> SimpleNamespace:
        assert module_name == (
            "bayescatrack.experiments.track2p_policy_confidence_ordered_strict_gap_cleanup"
        )
        return SimpleNamespace(main=lambda argv: calls.append(tuple(argv)) or 0)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    assert (
        cli.main(
            [
                "benchmark",
                "track2p-confidence-strict-gap-cleanup",
                "--data",
                "unused",
            ]
        )
        == 0
    )
    assert calls == [("--data", "unused")]
