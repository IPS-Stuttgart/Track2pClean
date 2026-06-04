"""Action-specific coherence suffix teacher-rescue benchmark preset.

The generic ``track2p-policy-coherence-suffix-teacher-rescue`` command exposes
many Track2p-teacher rescue knobs.  This thin wrapper fixes the safest residual
repair profile we have for the current Track2pPolicy-family lead:

* start from ComponentCleanup + coherence suffix stitching;
* use dynamic seed/cell confidence ordering;
* admit only target-extension or seed-source-backfill actions;
* use separate feature gates for target extensions and seed-source backfills;
* keep a tiny per-subject edit budget.

This is deliberately a candidate benchmark row, not a tuned default.  It mirrors
the checked-in ``benchmarks/coherence_teacher_action_specific.json`` probe and
makes that accuracy candidate directly executable as a Python module.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from bayescatrack.experiments.track2p_policy_coherence_suffix_teacher_rescue import (
    main as _main,
)

TRACK2P_POLICY_COHERENCE_SUFFIX_TEACHER_ACTION_SPECIFIC_METHOD = (
    "track2p-policy-coherence-suffix-teacher-action-specific"
)

_ARGS = (
    "--teacher-edge-order",
    "dynamic-seed-cell-confidence",
    "--teacher-action-filter",
    "target-extension-or-seed-source-backfill",
    "--teacher-feature-preset",
    "none",
    "--target-extension-feature-preset",
    "moderate-iou-cell-confidence",
    "--seed-source-feature-preset",
    "seed-source-cell-confident",
    "--no-allow-source-backfill",
    "--allow-seed-source-backfill",
    "--allow-completing-seed-source-backfill",
    "--min-teacher-component-observations",
    "2",
    "--max-applied-teacher-edits",
    "2",
)


def default_args() -> tuple[str, ...]:
    """Return the fixed action-specific teacher-rescue arguments."""

    return _ARGS


def main(argv: Sequence[str] | None = None) -> int:
    """Run the coherence suffix teacher-rescue command with fixed safe gates."""

    args = tuple(sys.argv[1:] if argv is None else argv)
    return int(_main([*_ARGS, *args]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
