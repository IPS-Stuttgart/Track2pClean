"""FullMHT runner with a complete-history terminal objective enabled.

This wrapper keeps the base scan-assignment implementation in
``track2p_policy_full_mht_benchmark`` and adds one label-free terminal reranking
term: incomplete seed-anchored histories.  It is meant as a focused method probe
for the paper question of whether complete-track-aware history selection can do
more than local pairwise assignment.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

METHOD = "track2p-policy-full-mht-terminal-completion"
_OPTION = "terminal_incomplete_history_weight"
_FLAG = "--terminal-incomplete-history-weight"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the terminal-completion FullMHT benchmark parser."""

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    parser = full_mht.build_arg_parser()
    parser.prog = f"bayescatrack benchmark {METHOD}"
    parser.description = (
        "Run FullMHT with an opt-in label-free terminal incomplete-history penalty."
    )
    _add_terminal_completion_arguments(parser)
    return parser


def _completion_only_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _add_terminal_completion_arguments(parser)
    return parser


def _add_terminal_completion_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("terminal complete-history objective")
    group.add_argument(
        _FLAG,
        dest=_OPTION,
        type=float,
        default=0.0,
        help=(
            "Penalty per missing observation in non-empty terminal histories. "
            "Default 0 preserves base FullMHT behavior."
        ),
    )


def _split_completion_args(argv: Sequence[str]) -> tuple[list[str], dict[str, float]]:
    """Return base FullMHT argv plus terminal completion config attrs."""

    namespace, base_argv = _completion_only_parser().parse_known_args(list(argv))
    return list(base_argv), {_OPTION: float(getattr(namespace, _OPTION))}


def _attach_completion_attrs(config: Any, attrs: dict[str, float]) -> Any:
    for key, value in attrs.items():
        object.__setattr__(config, key, value)
    return config


def main(argv: list[str] | None = None) -> int:
    """Run FullMHT with the terminal incomplete-history objective installed."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in raw_argv):
        build_arg_parser().parse_args(raw_argv)
        return 0

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht
    from bayescatrack.experiments.full_mht_terminal_completion_integration import (
        install_full_mht_terminal_completion_objective,
    )

    base_argv, completion_attrs = _split_completion_args(raw_argv)
    install_full_mht_terminal_completion_objective()

    original_config_class = full_mht.FullMHTConfig

    def full_mht_config_with_completion(*args: Any, **kwargs: Any) -> Any:
        return _attach_completion_attrs(
            original_config_class(*args, **kwargs), completion_attrs
        )

    full_mht.FullMHTConfig = full_mht_config_with_completion
    try:
        return int(full_mht.main(base_argv))
    finally:
        full_mht.FullMHTConfig = original_config_class


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
