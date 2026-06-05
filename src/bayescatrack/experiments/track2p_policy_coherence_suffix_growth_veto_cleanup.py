"""Apply growth-veto cleanup after CoherenceSuffixStitch.

This thin command wrapper runs the growth-veto cleanup machinery on the
CoherenceSuffixStitch prediction directly, without first applying Track2p
teacher-adjacent rescue.
"""

from __future__ import annotations

import sys

from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup


def main(argv: list[str] | None = None) -> int:
    """Run the non-teacher coherence-suffix growth-veto cleanup row."""

    args = list(sys.argv[1:] if argv is None else argv)
    if "--growth-veto-base" not in args:
        args.extend(["--growth-veto-base", "coherence-suffix"])
    return cleanup.main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
