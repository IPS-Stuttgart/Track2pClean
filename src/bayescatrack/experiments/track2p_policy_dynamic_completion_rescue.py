from __future__ import annotations

import sys
from collections.abc import Sequence

from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
    main as _main,
)

_ARGS = (
    "--allow-completing-rescue",
    "--allow-source-backfill",
    "--allow-seed-source-backfill",
    "--allow-fragment-merges",
    "--teacher-edge-order",
    "dynamic-confidence",
)


def default_args() -> tuple[str, ...]:
    return _ARGS


def main(argv: Sequence[str] | None = None) -> int:
    args = tuple(sys.argv[1:] if argv is None else argv)
    return int(_main([*_ARGS, *args]))


if __name__ == "__main__":
    raise SystemExit(main())
