"""Native Track2pClean command-line wrapper."""

from __future__ import annotations

import argparse
import importlib
import operator
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from bayescatrack import cli as _legacy_cli

_EXIT_CODE_ERROR = (
    "CLI delegates must return None or an integer exit code between 0 and 255"
)

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
    parser = _build_benchmark_help_parser()
    if not args:
        parser.print_help()
        return 0
    if args[0] in {"-h", "--help"}:
        parser.parse_args(args)
        return 0

    requested_command_name = args[0]
    command_name = (
        _legacy_cli._BENCHMARK_ALIASES.get(  # pylint: disable=protected-access
            requested_command_name, requested_command_name
        )
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
    program_name = f"track2pclean benchmark {requested_command_name}"
    legacy_program_name = f"bayescatrack benchmark {command_name}"
    return _handle_benchmark_module(
        command_args,
        module=module,
        program_name=program_name,
        legacy_program_name=legacy_program_name,
    )


def _handle_benchmark_module(
    args: list[str],
    *,
    module: Any,
    program_name: str,
    legacy_program_name: str,
) -> int:
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
        parser.prog = program_name
        return parser

    try:
        setattr(module, "build_arg_parser", _build_native_arg_parser)
        return _run_with_program_name(program_name, module.main, args)
    finally:
        setattr(module, "build_arg_parser", original_build_arg_parser)


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
        parser.prog = program_name
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


def _native_project_text(text: str) -> str:
    """Rewrite legacy project names in native Track2pClean CLI prose."""

    return text.replace("BayesCaTrack", "Track2pClean").replace(
        "bayescatrack",
        "track2pclean",
    )


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
    for attribute_name in ("description", "epilog", "usage"):
        value = getattr(parser, attribute_name, None)
        if isinstance(value, str):
            setattr(parser, attribute_name, value.replace(old_text, new_text))

    for group in (
        *getattr(parser, "_action_groups", ()),
        *getattr(parser, "_mutually_exclusive_groups", ()),
    ):
        for attribute_name in ("title", "description"):
            value = getattr(group, attribute_name, None)
            if isinstance(value, str):
                setattr(group, attribute_name, value.replace(old_text, new_text))

    for action in parser._actions:  # pylint: disable=protected-access
        help_text = getattr(action, "help", None)
        if isinstance(help_text, str):
            action.help = help_text.replace(old_text, new_text)

        metavar = getattr(action, "metavar", None)
        if isinstance(metavar, str):
            action.metavar = metavar.replace(old_text, new_text)
        elif isinstance(metavar, tuple):
            action.metavar = tuple(
                part.replace(old_text, new_text) if isinstance(part, str) else part
                for part in metavar
            )

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
        epilog="Includes diagnostics for Track2pClean edges.",
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=False)
    for (
        name,
        command,
    ) in _legacy_cli._BENCHMARK_COMMANDS.items():  # pylint: disable=protected-access
        subparsers.add_parser(name, help=_native_project_text(command.help))
    for (
        alias,
        canonical,
    ) in _legacy_cli._BENCHMARK_ALIASES.items():  # pylint: disable=protected-access
        subparsers.add_parser(
            alias, help=_native_project_text(f"Alias for {canonical}")
        )
    return parser


def _run_with_program_name(
    program_name: str,
    main_func: Callable[[list[str]], Any],
    args: list[str],
) -> int:
    previous_argv = sys.argv
    try:
        sys.argv = [program_name, *args]
        return _coerce_exit_code(main_func(args))
    finally:
        sys.argv = previous_argv


def _coerce_exit_code(result: Any) -> int:
    """Normalize delegated CLI return values to process exit codes."""

    if result is None:
        return 0
    if isinstance(result, np.ndarray):
        raise TypeError(_EXIT_CODE_ERROR)
    if isinstance(result, (bool, np.bool_)):
        raise TypeError(_EXIT_CODE_ERROR)
    try:
        exit_code = int(operator.index(result))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(_EXIT_CODE_ERROR) from exc
    if not 0 <= exit_code <= 255:
        raise ValueError(_EXIT_CODE_ERROR)
    return exit_code


__all__ = ["main"]
