"""Native Track2pClean command-line wrapper."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from typing import Any

from bayescatrack import cli as _legacy_cli

_TOP_LEVEL_HELP = """usage: track2pclean {summary,export,benchmark,growth,advanced} ...

Track2pClean command line tools.

commands:
  summary       Print a JSON summary for one Track2p-style subject directory.
  export        Export a PyRecEst-ready NPZ bundle for one subject.
  benchmark     Run reproducible benchmark harnesses.
  growth        Analyze global growth/displacement patterns from track tables.
  advanced      Advanced diagnostics and result-improvement workbench helpers.

Run 'track2pclean <command> --help' for command-specific options.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch Track2pClean CLI commands with native program names."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_TOP_LEVEL_HELP)
        return 0

    if args[0] == "benchmark":
        return _handle_benchmark(args[1:])

    return _run_with_program_name("track2pclean", _legacy_cli.main, args)


def _handle_benchmark(args: list[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        _build_benchmark_help_parser().parse_args(args)
        return 0

    command_name = _legacy_cli._BENCHMARK_ALIASES.get(  # pylint: disable=protected-access
        args[0], args[0]
    )
    command = _legacy_cli._BENCHMARK_COMMANDS.get(  # pylint: disable=protected-access
        command_name
    )
    if command is None:
        parser = argparse.ArgumentParser(prog="track2pclean benchmark")
        parser.error(f"unknown benchmark {args[0]!r}")
        return 2

    module = importlib.import_module(command.module)
    return _run_with_program_name(
        f"track2pclean benchmark {command_name}",
        module.main,
        args[1:],
    )


def _build_benchmark_help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="track2pclean benchmark",
        description="Run Track2pClean benchmark harnesses.",
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=False)
    for name, command in _legacy_cli._BENCHMARK_COMMANDS.items():  # pylint: disable=protected-access
        subparsers.add_parser(name, help=command.help)
    for alias, canonical in _legacy_cli._BENCHMARK_ALIASES.items():  # pylint: disable=protected-access
        subparsers.add_parser(alias, help=f"Alias for {canonical}")
    return parser


def _run_with_program_name(
    program_name: str,
    main_func: Callable[[list[str]], Any],
    args: list[str],
) -> int:
    previous_argv = sys.argv
    try:
        sys.argv = [program_name, *args]
        return int(main_func(args))
    finally:
        sys.argv = previous_argv


__all__ = ["main"]
