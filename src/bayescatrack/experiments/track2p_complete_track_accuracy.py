"""Complete-track-F1-oriented Track2p accuracy preset.

This module is a thin, reproducible preset around the Track2p-policy
component-cleanup sweep. The generic sweep is intentionally flexible; this
preset encodes the paper-facing operating point we most often want when the
primary target is longitudinal identity continuity: select the best candidate by
complete-track micro-F1 and write only that selected candidate by default.
"""

from __future__ import annotations

from collections.abc import Sequence

from bayescatrack.experiments import track2p_policy_component_sweep as _component_sweep

DEFAULT_SPLIT_RISK_THRESHOLDS = "0.75,1.0,1.25,1.5,1.75,2.0,2.5"
DEFAULT_SPLIT_PENALTIES = "0.0,0.1,0.25,0.5,1.0"
DEFAULT_MIN_SIDE_OBSERVATIONS = "2,3"
DEFAULT_REQUIRE_COMPLETE_TRACK_OPTIONS = "true,false"
DEFAULT_OBJECTIVE = "complete_track_f1_micro"

_PRESET_ARGUMENT_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("--split-risk-thresholds", DEFAULT_SPLIT_RISK_THRESHOLDS),
    ("--split-penalties", DEFAULT_SPLIT_PENALTIES),
    ("--min-side-observations", DEFAULT_MIN_SIDE_OBSERVATIONS),
    ("--require-complete-track-options", DEFAULT_REQUIRE_COMPLETE_TRACK_OPTIONS),
    ("--objective", DEFAULT_OBJECTIVE),
)


def main(argv: list[str] | None = None) -> int:
    """Run the complete-track accuracy preset CLI."""

    args = list(argv or [])
    all_candidates = _pop_flag(args, "--all-candidates")
    for option, value in _PRESET_ARGUMENT_DEFAULTS:
        if not _contains_option(args, option):
            args.extend((option, value))
    if not all_candidates and not _contains_option(args, "--best-only"):
        args.append("--best-only")
    return int(_component_sweep.main(args))


def _pop_flag(args: list[str], flag: str) -> bool:
    found = False
    retained: list[str] = []
    for arg in args:
        if arg == flag:
            found = True
        else:
            retained.append(arg)
    args[:] = retained
    return found


def _contains_option(args: Sequence[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
