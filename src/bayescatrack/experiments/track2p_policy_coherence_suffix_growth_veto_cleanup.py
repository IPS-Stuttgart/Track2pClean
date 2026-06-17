"""Apply growth-veto cleanup after CoherenceSuffixStitch.

This thin command wrapper runs the growth-veto cleanup machinery on the
CoherenceSuffixStitch prediction directly, without first applying Track2p
teacher-adjacent rescue.
"""

from __future__ import annotations

import sys

from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup


def _option_present(args: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in args)


def _with_coherence_suffix_default(args: list[str]) -> list[str]:
    if _option_present(args, "--growth-veto-base"):
        return list(args)
    return [*args, "--growth-veto-base", "coherence-suffix"]


def main(argv: list[str] | None = None) -> int:
    """Run the non-teacher coherence-suffix growth-veto cleanup row."""

    args = list(sys.argv[1:] if argv is None else argv)
    return cleanup.main(_with_coherence_suffix_default(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
