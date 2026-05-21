"""Align Track2p cost-sweep Suite2p defaults with benchmark defaults."""

from __future__ import annotations

import argparse
from typing import Any


def install_cost_sweep_suite2p_defaults() -> None:
    """Patch cost-sweep parser/config defaults idempotently."""

    from bayescatrack.experiments import track2p_cost_sweep as cost_sweep

    original_build_arg_parser = cost_sweep.build_arg_parser
    if getattr(original_build_arg_parser, "_bayescatrack_cost_sweep_defaults", False):
        return

    def _build_arg_parser_with_suite2p_defaults() -> argparse.ArgumentParser:
        parser = original_build_arg_parser()
        _set_parser_default(parser, "include_non_cells", True)
        return parser

    setattr(
        _build_arg_parser_with_suite2p_defaults,
        "_bayescatrack_cost_sweep_defaults",
        True,
    )
    setattr(
        _build_arg_parser_with_suite2p_defaults,
        "_bayescatrack_original",
        original_build_arg_parser,
    )
    cost_sweep.build_arg_parser = _build_arg_parser_with_suite2p_defaults


def _set_parser_default(
    parser: argparse.ArgumentParser,
    dest: str,
    value: Any,
) -> None:
    parser.set_defaults(**{dest: value})
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == dest:
            action.default = value
            return
