"""PyRecEst residual-MHT with a TP-safe relaxed growth-veto frontier.

The first relaxed frontier-MHT run selected the desired false-positive terminal
edge, but it also selected a true-positive terminal edge. This wrapper keeps
the residual-MHT candidate set broader than the strict one-edge growth veto in
terms of growth residual and row/column rank, while restoring the local-confidence
caps that protect strong terminal continuations:

* moderate, not very high, registered/shifted overlap;
* weak-to-moderate endpoint cell probability;
* terminal/last-session complete-component structure;
* at most two residual edits per subject.

It is still a residual edit-level MHT, not a global all-ROI MHT tracker.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from bayescatrack.experiments import (
    track2p_policy_pyrecest_residual_mht_cleanup as residual_mht,
)

_DEFAULT_OPTION_VALUES: tuple[tuple[str, str], ...] = (
    # Broaden growth/rank frontier relative to strict growth veto.
    ("--min-growth-residual-mahalanobis", "12"),
    ("--min-growth-residual", "2.0"),
    ("--min-veto-registered-iou", "0.35"),
    ("--min-veto-shifted-iou", "0.45"),
    ("--max-veto-row-rank", "2"),
    ("--max-veto-column-rank", "2"),
    # Restore conservative local-confidence caps; the unsafe frontier TP looked
    # plausible locally, while the known FP veto sits in the moderate-overlap
    # and weak-endpoint pocket.
    ("--max-veto-registered-iou", "0.60"),
    ("--max-veto-shifted-iou", "0.80"),
    ("--min-veto-cell-probability", "0.50"),
    ("--max-veto-min-cell-probability", "0.65"),
    ("--max-veto-local-neighbor-distortion", "none"),
    # Keep structural veto constraints.
    ("--require-veto-not-suffix-edge", ""),
    ("--require-veto-terminal-edge", ""),
    ("--require-veto-last-session-edge", ""),
    ("--require-veto-complete-component", ""),
    # Let PyRecEst see a small frontier, but make extra edits pay a stronger
    # penalty than in the first relaxed run.
    ("--mht-candidate-top-k", "8"),
    ("--mht-max-edits-per-subject", "2"),
    ("--mht-max-hypotheses", "32"),
    ("--mht-edit-penalty", "0.55"),
    ("--mht-score-threshold", "1.6"),
)


def _option_present(args: Sequence[str], option: str) -> bool:
    """Return true when ``option`` is already provided by the user."""

    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in args)


def _with_default_options(args: Sequence[str]) -> list[str]:
    """Prepend wrapper defaults while allowing user arguments to override them."""

    output: list[str] = []
    for option, value in _DEFAULT_OPTION_VALUES:
        if _option_present(args, option):
            continue
        output.append(option)
        if value:
            output.append(value)
    output.extend(args)
    return output


def main(argv: list[str] | None = None) -> int:
    """Run the TP-safe frontier residual-MHT wrapper."""

    args = list(sys.argv[1:] if argv is None else argv)
    return residual_mht.main(_with_default_options(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
