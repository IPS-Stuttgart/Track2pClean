from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

from bayescatrack._cli_exit_code_validation import install_cli_exit_code_validation


def _make_fake_cli_module() -> ModuleType:
    cli_module = ModuleType("tests.fake_bayescatrack_cli")
    cli_module._TOP_LEVEL_HELP = "help"
    cli_module._BENCHMARK_ALIASES = {}
    cli_module._BENCHMARK_COMMANDS = {}

    def main(argv: list[str] | None = None) -> Any:
        return None

    def handle_benchmark(args: list[str]) -> Any:
        return None

    def core_main(args: list[str]) -> Any:
        return None

    def build_benchmark_help_parser() -> SimpleNamespace:
        return SimpleNamespace(parse_args=lambda args: None)

    cli_module.main = main  # type: ignore[attr-defined]
    cli_module._handle_benchmark = handle_benchmark  # type: ignore[attr-defined]
    cli_module._core_main = core_main  # type: ignore[attr-defined]
    cli_module._build_benchmark_help_parser = build_benchmark_help_parser  # type: ignore[attr-defined]
    return cli_module


def test_exit_code_validation_repatches_when_stale_flag_outlives_wrappers() -> None:
    cli_module = _make_fake_cli_module()
    cli_module._track2pclean_exit_code_validation_installed = True  # type: ignore[attr-defined]

    install_cli_exit_code_validation(cli_module)

    assert cli_module.main(["summary"]) == 0  # type: ignore[attr-defined]
    assert cli_module._track2pclean_exit_code_validation_installed is True  # type: ignore[attr-defined]


def test_exit_code_validation_install_is_idempotent_for_marked_wrappers() -> None:
    cli_module = _make_fake_cli_module()

    install_cli_exit_code_validation(cli_module)
    first_main = cli_module.main  # type: ignore[attr-defined]
    first_handle_benchmark = cli_module._handle_benchmark  # type: ignore[attr-defined]
    install_cli_exit_code_validation(cli_module)

    assert cli_module.main is first_main  # type: ignore[attr-defined]
    assert cli_module._handle_benchmark is first_handle_benchmark  # type: ignore[attr-defined]
