"""Apply growth-veto cleanup after CoherenceSuffixStitch.

This thin command wrapper runs the growth-veto cleanup machinery on the
CoherenceSuffixStitch prediction directly, without first applying Track2p
teacher-adjacent rescue.
"""

from __future__ import annotations

import sys

from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup


def _option_values(args: list[str], option: str) -> list[str]:
    prefix = f"{option}="
    values: list[str] = []
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            values.append(arg.split("=", 1)[1])
        if arg == option:
            if index + 1 >= len(args):
                values.append("")
            else:
                values.append(args[index + 1])
    return values


def main(argv: list[str] | None = None) -> int:
    """Run the non-teacher coherence-suffix growth-veto cleanup row."""

    args = list(sys.argv[1:] if argv is None else argv)
    growth_veto_bases = _option_values(args, "--growth-veto-base")
    if not growth_veto_bases:
        args.extend(["--growth-veto-base", "coherence-suffix"])
    elif any(base != "coherence-suffix" for base in growth_veto_bases):
        raise SystemExit(
            "track2p-policy-coherence-suffix-growth-veto-cleanup requires "
            "--growth-veto-base coherence-suffix."
        )
    return cleanup.main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
