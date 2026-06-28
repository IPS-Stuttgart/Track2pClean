from __future__ import annotations

import sys
from types import ModuleType

from bayescatrack import cli as bayescatrack_cli
from bayescatrack._cli_exit_code_validation import install_cli_exit_code_validation
from tests import _support  # noqa: F401


def test_cli_exit_code_validation_reinstalls_after_stale_installed_flag(monkeypatch):
    def stale_main(argv=None):
        return int(None)

    def stale_handle_benchmark(args):
        return int(None)

    monkeypatch.setattr(bayescatrack_cli, "main", stale_main)
    monkeypatch.setattr(bayescatrack_cli, "_handle_benchmark", stale_handle_benchmark)
    monkeypatch.setattr(bayescatrack_cli, "_coerce_exit_code", lambda value: int(value))
    monkeypatch.setattr(
        bayescatrack_cli,
        "_track2pclean_exit_code_validation_installed",
        True,
        raising=False,
    )

    install_cli_exit_code_validation(bayescatrack_cli)

    assert bayescatrack_cli.main is not stale_main
    assert bayescatrack_cli._handle_benchmark is not stale_handle_benchmark
    monkeypatch.setattr(bayescatrack_cli, "_core_main", lambda args: None)
    assert bayescatrack_cli.main(["summary", "--example"]) == 0


def test_reinstalled_benchmark_handler_normalizes_none_delegate(monkeypatch):
    fake_module = ModuleType("tests.reinstalled_none_returning_benchmark")

    def fake_main(args):
        assert args == ["--example"]
        return None

    fake_module.main = fake_main
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setitem(
        bayescatrack_cli._BENCHMARK_COMMANDS,
        "reinstalled-none-returning",
        bayescatrack_cli._BenchmarkCommand(
            fake_module.__name__,
            "No-op benchmark",
        ),
    )

    install_cli_exit_code_validation(bayescatrack_cli)

    assert (
        bayescatrack_cli._handle_benchmark(
            ["reinstalled-none-returning", "--example"],
        )
        == 0
    )
