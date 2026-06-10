"""Apply growth-veto cleanup after CoherenceSuffixStitch.

This thin command wrapper runs the growth-veto cleanup machinery on the
CoherenceSuffixStitch prediction directly, without first applying Track2p
teacher-adjacent rescue.
"""

from __future__ import annotations

import sys

from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup


def _option_value(args: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
        if arg == option:
            if index + 1 >= len(args):
                return ""
            return args[index + 1]
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the non-teacher coherence-suffix growth-veto cleanup row."""

    args = list(sys.argv[1:] if argv is None else argv)
    growth_veto_base = _option_value(args, "--growth-veto-base")
    if growth_veto_base is None:
        args.extend(["--growth-veto-base", "coherence-suffix"])
    elif growth_veto_base != "coherence-suffix":
        raise SystemExit(
            "track2p-policy-coherence-suffix-growth-veto-cleanup requires "
            "--growth-veto-base coherence-suffix."
        )
    return cleanup.main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
