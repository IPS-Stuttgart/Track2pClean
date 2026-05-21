"""QA CLI Suite2p default integration."""

from __future__ import annotations

import argparse
from typing import Any


def install_qa_suite2p_defaults() -> None:
    """Patch QA parser defaults idempotently."""

    from bayescatrack.experiments import registration_qa_report as qa_report

    original_build_arg_parser = qa_report.build_arg_parser
    if getattr(original_build_arg_parser, "_bayescatrack_qa_defaults", False):
        return

    def _build_arg_parser_with_defaults() -> argparse.ArgumentParser:
        parser = original_build_arg_parser()
        _set_parser_default(parser, "include_non_cells", True)
        _add_store_false_alias_if_missing(
            parser,
            option="--no-include-non-cells",
            dest="include_non_cells",
            help_text="Hard-filter Suite2p ROIs using iscell.npy.",
        )
        return parser

    setattr(_build_arg_parser_with_defaults, "_bayescatrack_qa_defaults", True)
    setattr(_build_arg_parser_with_defaults, "_bayescatrack_original", original_build_arg_parser)
    qa_report.build_arg_parser = _build_arg_parser_with_defaults


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
