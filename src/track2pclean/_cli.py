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
    if args[0] == "growth":
        return _handle_module_command(
            args[1:],
            module_name="bayescatrack.analysis.growth",
            program_name="track2pclean growth",
            legacy_program_name="bayescatrack growth",
        )
    if args[0] == "advanced":
        return _handle_module_command(
            args[1:],
            module_name="bayescatrack.experiments.advanced_improvement_workbench",
            program_name="track2pclean advanced",
            legacy_program_name="python -m bayescatrack.experiments.advanced_improvement_workbench",
        )

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
    command_args = args[1:]
    if command_args and command_args[0] in {"-h", "--help"} and hasattr(
        module, "build_arg_parser"
    ):
        parser = module.build_arg_parser()
        parser.prog = f"track2pclean benchmark {command_name}"
        parser.parse_args(command_args)
        return 0
    return _run_with_program_name(
        f"track2pclean benchmark {command_name}",
        module.main,
        command_args,
    )


def _handle_module_command(
    args: list[str],
    *,
    module_name: str,
    program_name: str,
    legacy_program_name: str,
) -> int:
    module = importlib.import_module(module_name)
    original_build_arg_parser = getattr(module, "build_arg_parser", None)

    if not callable(original_build_arg_parser):
        return _run_with_program_name(program_name, module.main, args)

    def _build_native_arg_parser(
        *parser_args: Any,
        **parser_kwargs: Any,
    ) -> argparse.ArgumentParser:
        parser = original_build_arg_parser(*parser_args, **parser_kwargs)
        _retitle_arg_parser(
            parser,
            legacy_program_name=legacy_program_name,
            program_name=program_name,
        )
        return parser

    try:
        setattr(module, "build_arg_parser", _build_native_arg_parser)
        return _run_with_program_name(program_name, module.main, args)
    finally:
        setattr(module, "build_arg_parser", original_build_arg_parser)


def _retitle_arg_parser(
    parser: argparse.ArgumentParser,
    *,
    legacy_program_name: str,
    program_name: str,
) -> None:
    _replace_parser_program_prefix(
        parser,
        legacy_program_name=legacy_program_name,
        program_name=program_name,
    )
    _replace_parser_text(parser, "BayesCaTrack", "Track2pClean")
    _replace_parser_text(parser, "bayescatrack", "track2pclean")


def _replace_parser_program_prefix(
    parser: argparse.ArgumentParser,
    *,
    legacy_program_name: str,
    program_name: str,
) -> None:
    if parser.prog == legacy_program_name or parser.prog.startswith(
        f"{legacy_program_name} "
    ):
        parser.prog = f"{program_name}{parser.prog[len(legacy_program_name):]}"

    for child_parser in _iter_child_arg_parsers(parser):
        _replace_parser_program_prefix(
            child_parser,
            legacy_program_name=legacy_program_name,
            program_name=program_name,
        )


def _replace_parser_text(
    parser: argparse.ArgumentParser,
    old_text: str,
    new_text: str,
) -> None:
    for attribute_name in ("description", "epilog"):
        value = getattr(parser, attribute_name, None)
        if isinstance(value, str):
            setattr(parser, attribute_name, value.replace(old_text, new_text))

    for action in parser._actions:  # pylint: disable=protected-access
        help_text = getattr(action, "help", None)
        if isinstance(help_text, str):
            action.help = help_text.replace(old_text, new_text)

    for child_parser in _iter_child_arg_parsers(parser):
        _replace_parser_text(child_parser, old_text, new_text)


def _iter_child_arg_parsers(
    parser: argparse.ArgumentParser,
) -> list[argparse.ArgumentParser]:
    child_parsers: list[argparse.ArgumentParser] = []
    for action in parser._actions:  # pylint: disable=protected-access
        choices = getattr(action, "choices", None)
        if not isinstance(choices, dict):
            continue
        child_parsers.extend(
            choice
            for choice in choices.values()
            if isinstance(choice, argparse.ArgumentParser)
        )
    return child_parsers


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
