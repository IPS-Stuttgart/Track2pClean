"""Guarded Track2p-policy gap-consensus sweep entry point.

This wrapper keeps the existing gap-consensus sweep implementation but changes the
paper-facing default grid to compare adjacent-only consensus cleanup (`max_gap=1`)
against gap-rescue cleanup. The aggregate ranker already uses pairwise F1 as a
tie-breaker after the requested objective, so including the adjacent-only point
lets the sweep avoid gap-rescue false positives when complete-track F1 is tied.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    ThresholdMethod,
)
from bayescatrack.experiments.track2p_policy_gap_consensus_cleanup import (
    TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP,
)
from bayescatrack.experiments.track2p_policy_gap_consensus_sweep import (
    GapConsensusSweepConfig,
    GapConsensusSweepOutput,
)
from bayescatrack.experiments.track2p_policy_gap_consensus_sweep import (
    build_arg_parser as _build_sweep_arg_parser,
)
from bayescatrack.experiments.track2p_policy_gap_consensus_sweep import (
    main as _sweep_main,
)
from bayescatrack.experiments.track2p_policy_gap_consensus_sweep import (
    run_track2p_policy_gap_consensus_sweep,
)

GUARDED_GAP_CONSENSUS_DEFAULT_MAX_GAPS: tuple[int, ...] = tuple(
    dict.fromkeys((1, TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP))
)


def guarded_gap_consensus_sweep_config(**overrides: Any) -> GapConsensusSweepConfig:
    """Build the guarded default sweep config.

    Unless explicitly overridden, the grid evaluates both adjacent-only cleanup
    and the default gap-rescue setting. This makes complete-track-F1-oriented
    sweeps robust to the empirically observed case where gap rescue keeps the
    same complete-track F1 but adds many pairwise false positives.
    """

    values = dict(overrides)
    values.setdefault("max_gaps", GUARDED_GAP_CONSENSUS_DEFAULT_MAX_GAPS)
    return GapConsensusSweepConfig(**values)


def run_track2p_policy_gap_consensus_guarded_sweep(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    sweep_config: GapConsensusSweepConfig | None = None,
) -> GapConsensusSweepOutput:
    """Run the gap-consensus sweep with guarded max-gap defaults."""

    return run_track2p_policy_gap_consensus_sweep(
        config,
        threshold_method=threshold_method,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
        sweep_config=sweep_config or guarded_gap_consensus_sweep_config(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a CLI parser that advertises the guarded max-gap default."""

    parser = _build_sweep_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-policy-gap-consensus-guarded-sweep"
    parser.description = (
        "Sweep Track2p-policy consensus cleanup settings with guarded max-gap "
        "defaults that include adjacent-only and gap-rescue candidates."
    )
    _set_option_default(parser, "--max-gaps", _guarded_max_gaps_arg())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guarded sweep CLI."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(args, "--max-gaps"):
        args.extend(("--max-gaps", _guarded_max_gaps_arg()))
    return int(_sweep_main(args))


def _guarded_max_gaps_arg() -> str:
    return ",".join(str(value) for value in GUARDED_GAP_CONSENSUS_DEFAULT_MAX_GAPS)


def _has_option(args: Sequence[str], option: str) -> bool:
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in args)


def _set_option_default(
    parser: argparse.ArgumentParser, option: str, value: str
) -> None:
    # pylint: disable=protected-access
    action = parser._option_string_actions.get(option)
    if action is None:  # pragma: no cover - defensive against upstream parser changes
        raise RuntimeError(f"could not find parser option {option!r}")
    action.default = value


__all__ = (
    "GUARDED_GAP_CONSENSUS_DEFAULT_MAX_GAPS",
    "build_arg_parser",
    "guarded_gap_consensus_sweep_config",
    "main",
    "run_track2p_policy_gap_consensus_guarded_sweep",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
