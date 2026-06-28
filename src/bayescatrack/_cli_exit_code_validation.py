"""Normalize BayesCaTrack CLI delegate return values."""

from __future__ import annotations

import argparse
import importlib
import operator
import sys
from typing import Any

import numpy as np

# pylint: disable=protected-access


_EXIT_CODE_ERROR = "CLI delegates must return None or an integer exit code"


def install_cli_exit_code_validation(cli_module: Any) -> None:
    """Patch ``bayescatrack.cli`` so delegate return values become exit codes."""

    if getattr(cli_module, "_track2pclean_exit_code_validation_installed", False):
        return

    def main(argv: list[str] | None = None) -> int:
        """Dispatch BayesCaTrack CLI commands with normalized exit codes."""

        args = list(sys.argv[1:] if argv is None else argv)
        if not args or args[0] in {"-h", "--help"}:
            print(cli_module._TOP_LEVEL_HELP)
            return 0

        if args[0] == "growth":
            from bayescatrack.analysis.growth import main as _growth_main

            return _coerce_exit_code(_growth_main(args[1:]))

        if args[0] == "advanced":
            from bayescatrack.experiments.advanced_improvement_workbench import (
                main as _advanced_main,
            )

            return _coerce_exit_code(_advanced_main(args[1:]))

        if args[0] != "benchmark":
            return _coerce_exit_code(cli_module._core_main(args))

        return handle_benchmark(args[1:])

    def handle_benchmark(args: list[str]) -> int:
        """Dispatch BayesCaTrack benchmark subcommands with normalized exit codes."""

        if not args or args[0] in {"-h", "--help"}:
            cli_module._build_benchmark_help_parser().parse_args(args)
            return 0

        command_name = cli_module._BENCHMARK_ALIASES.get(args[0], args[0])
        command = cli_module._BENCHMARK_COMMANDS.get(command_name)
        if command is None:
            parser = argparse.ArgumentParser(prog="bayescatrack benchmark")
            parser.error(f"unknown benchmark {args[0]!r}")
            return 2

        module = importlib.import_module(command.module)
        return _coerce_exit_code(module.main(args[1:]))

    cli_module.main = main
    cli_module._handle_benchmark = handle_benchmark
    cli_module._coerce_exit_code = _coerce_exit_code
    cli_module._track2pclean_exit_code_validation_installed = True


def _coerce_exit_code(result: Any) -> int:
    """Normalize delegated CLI return values to process exit codes."""

    if result is None:
        return 0
    if isinstance(result, (bool, np.bool_)):
        raise TypeError(_EXIT_CODE_ERROR)
    try:
        return int(operator.index(result))
    except TypeError as exc:
        raise TypeError(_EXIT_CODE_ERROR) from exc


__all__ = ["install_cli_exit_code_validation"]
