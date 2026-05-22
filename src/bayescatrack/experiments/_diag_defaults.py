"""Diagnostic CLI ROI default integration."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any


def install_diagnostic_suite2p_defaults() -> None:
    """Patch diagnostic parser defaults idempotently."""

    from bayescatrack.experiments import registration_qa_report
    from bayescatrack.experiments import track2p_teacher_audit

    _patch_parser_factory(registration_qa_report, marker="_bayescatrack_registration_qa_defaults")
    _patch_parser_factory(track2p_teacher_audit, marker="_bayescatrack_teacher_defaults")


def _patch_parser_factory(module: Any, *, marker: str) -> None:
    original_build_arg_parser: Callable[[], argparse.ArgumentParser] = module.build_arg_parser
    if getattr(original_build_arg_parser, marker, False):
        return

    def _build_arg_parser_with_defaults() -> argparse.ArgumentParser:
        parser = original_build_arg_parser()
        _set_parser_default(parser, "include_non_cells", True)
        _add_store_false_alias_if_missing(
            parser,
            option="--no-include-non-cells",
            dest="include_non_cells",
            help_text="Use Suite2p cell filtering.",
        )
        return parser

    setattr(_build_arg_parser_with_defaults, marker, True)
    setattr(_build_arg_parser_with_defaults, "_bayescatrack_original", original_build_arg_parser)
    module.build_arg_parser = _build_arg_parser_with_defaults


def _set_parser_default(parser: argparse.ArgumentParser, dest: str, value: Any) -> None:
    parser.set_defaults(**{dest: value})
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == dest:
            action.default = value
            return


def _add_store_false_alias_if_missing(
    parser: argparse.ArgumentParser,
    *,
    option: str,
    dest: str,
    help_text: str,
) -> None:
    if any(option in action.option_strings for action in parser._actions):  # pylint: disable=protected-access
        return
    parser.add_argument(option, dest=dest, action="store_false", help=help_text)
