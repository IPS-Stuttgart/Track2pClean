"""Relaxed/frontier PyRecEst residual-MHT cleanup row.

The conservative PyRecEst residual-MHT row is intentionally close to the
hand-coded CoherenceSuffixGrowthVeto gate and therefore tends to collapse to the
same single safe veto. This wrapper pushes the MHT formulation one step harder
without opening a broad all-ROI tracker:

* start from the non-teacher CoherenceSuffixStitch state;
* keep the candidate family restricted to splittable terminal complete-component
  growth-veto hypotheses;
* relax the growth/overlap/rank gates enough to expose a small frontier;
* let PyRecEst's residual-MHT selector choose at most a tiny edit set;
* keep all GT labels and score deltas audit-only in the underlying runner.

This command deliberately changes only defaults. Any option supplied explicitly
on the command line wins over the frontier preset.
"""

from __future__ import annotations

import sys

from bayescatrack.experiments import (
    track2p_policy_pyrecest_residual_mht_cleanup as residual_mht,
)

# The preset is conservative relative to a full sweep, but broader than the
# strict growth-veto pocket:
# - lower Mahalanobis/residual thresholds;
# - allow rank-2 local candidates;
# - allow a wider moderate-overlap window;
# - allow two selected edits per subject, but with a larger edit penalty.
_FRONTIER_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("--min-growth-residual-mahalanobis", "12"),
    ("--min-growth-residual", "2.0"),
    ("--min-veto-registered-iou", "0.35"),
    ("--max-veto-registered-iou", "0.75"),
    ("--min-veto-shifted-iou", "0.45"),
    ("--max-veto-shifted-iou", "0.90"),
    ("--min-veto-cell-probability", "0.50"),
    ("--max-veto-min-cell-probability", "0.80"),
    ("--max-veto-local-neighbor-distortion", "none"),
    ("--max-veto-row-rank", "2"),
    ("--max-veto-column-rank", "2"),
    ("--mht-candidate-top-k", "8"),
    ("--mht-max-edits-per-subject", "2"),
    ("--mht-max-hypotheses", "32"),
    ("--mht-edit-penalty", "0.40"),
    ("--mht-score-threshold", "1.40"),
)

_BOOLEAN_DEFAULTS: tuple[str, ...] = (
    "--require-veto-not-suffix-edge",
    "--require-veto-terminal-edge",
    "--require-veto-last-session-edge",
    "--require-veto-complete-component",
)


def _option_present(args: list[str], option: str) -> bool:
    """Return whether ``option`` or an explicit ``--no-`` form is present."""

    if option in args:
        return True
    if option.startswith("--"):
        negative = "--no-" + option[2:]
        if negative in args:
            return True
    return False


def _with_frontier_defaults(args: list[str]) -> list[str]:
    """Append frontier defaults for options absent from ``args``."""

    output = list(args)
    for option, value in _FRONTIER_DEFAULTS:
        if not _option_present(output, option):
            output.extend([option, value])
    for option in _BOOLEAN_DEFAULTS:
        if not _option_present(output, option):
            output.append(option)
    return output


def main(argv: list[str] | None = None) -> int:
    """Run the relaxed/frontier PyRecEst residual-MHT cleanup row."""

    args = list(sys.argv[1:] if argv is None else argv)
    return residual_mht.main(_with_frontier_defaults(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
