"""Decision helper for FullMHT scan-history dynamics probe manifests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bayescatrack.experiments.full_mht_history_dynamics_decision import (
    HistoryDynamicsDecisionConfig,
    evaluate_history_dynamics_decision,
    format_decision_markdown,
    load_comparison_rows,
)

SCAN_HISTORY_DYNAMICS_CONFIG = HistoryDynamicsDecisionConfig(
    baseline="FullMHTPrior2",
    candidates=(
        "FullMHTScanHistoryDynamics025",
        "FullMHTScanHistoryDynamics050",
        "FullMHTScanHistoryDynamics100",
    ),
    track2p="Track2p",
)


def evaluate_scan_history_dynamics_decision(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate whether scan-time history dynamics is stable enough to promote."""

    return evaluate_history_dynamics_decision(
        rows,
        config=SCAN_HISTORY_DYNAMICS_CONFIG,
    )


def write_scan_history_decision(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write a scan-history decision artifact as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(format_decision_markdown(decision) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_scan_history_dynamics_decision",
        description="Interpret a FullMHT scan-history dynamics comparison CSV.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decision = evaluate_scan_history_dynamics_decision(
        load_comparison_rows(args.comparison_csv)
    )
    if args.output is not None:
        write_scan_history_decision(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_decision_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
