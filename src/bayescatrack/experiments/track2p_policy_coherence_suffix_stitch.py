"""Promoted coherence-gated suffix stitch benchmark row."""

from __future__ import annotations

import argparse

from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_stitch_whatif as _whatif,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the promoted suffix-stitch row."""

    parser = _whatif.build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-policy-coherence-suffix-stitch"
    parser.description = "Run ComponentCleanup plus coherence-gated suffix stitching."
    parser.set_defaults(aggregate_row=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the promoted coherence suffix-stitch benchmark row."""

    return int(_whatif.main(argv, parser=build_arg_parser()))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
