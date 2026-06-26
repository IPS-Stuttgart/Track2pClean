"""Decision helper for FullMHT growth-history prediction probe manifests.

The growth-history prediction layer applies a row-history dynamics penalty while
scan assignment costs are built.  This helper freezes how to interpret the probe:
a candidate is interesting only if it improves complete-track F1 without
pairwise-F1 loss across more than one nearby weight.
"""

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

GROWTH_HISTORY_PREDICTION_CONFIG = HistoryDynamicsDecisionConfig(
    baseline="FullMHTPrior2",
    candidates=(
        "FullMHTGrowthHistoryPrediction025",
        "FullMHTGrowthHistoryPrediction050",
        "FullMHTGrowthHistoryPrediction100",
    ),
    track2p="Track2p",
)


def evaluate_growth_history_prediction_decision(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate whether growth-history prediction is stable enough to promote."""

    decision = evaluate_history_dynamics_decision(
        rows,
        config=GROWTH_HISTORY_PREDICTION_CONFIG,
    )
    if decision.get("status") == "complete":
        decision = dict(decision)
        decision["growth_history_prediction_result"] = decision.pop(
            "history_dynamics_result"
        )
        decision["recommendation"] = _growth_prediction_recommendation(
            str(decision["growth_history_prediction_result"])
        )
    return decision


def format_growth_history_prediction_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact growth-history prediction decision note."""

    if decision.get("status") != "complete":
        missing = ", ".join(str(item) for item in decision.get("missing_approaches", ()))
        return "\n".join(
            [
                "# FullMHT Growth-History Prediction Decision",
                "",
                "Status: incomplete",
                f"Missing approaches: {missing or 'none reported'}",
                f"Recommendation: {decision.get('recommendation', '')}",
            ]
        )

    generic = format_decision_markdown(
        {
            **dict(decision),
            "history_dynamics_result": decision["growth_history_prediction_result"],
        }
    )
    return generic.replace(
        "# FullMHT History Dynamics Decision",
        "# FullMHT Growth-History Prediction Decision",
        1,
    )


def write_growth_history_prediction_decision(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write a growth-history prediction decision artifact as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(
        format_growth_history_prediction_markdown(decision) + "\n",
        encoding="utf-8",
    )


def _growth_prediction_recommendation(result: str) -> str:
    if result == "history_dynamics_stable_gain":
        return (
            "promote only after exposure/no-GT checks; this is a promising "
            "scan-assignment dynamics component"
        )
    if result == "history_dynamics_single_weight_gain":
        return "treat as exploratory; gain appears knife-edge"
    if result == "history_dynamics_ties_baseline":
        return "record as dynamics-equivalent to the proposal-prior control"
    if result == "history_dynamics_pairwise_regression":
        return "do not promote; dynamics term damages pairwise tracking"
    if result == "history_dynamics_complete_regression":
        return "do not promote; dynamics term damages complete-track identity"
    return "rerun or inspect probe outputs before interpreting"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_growth_history_prediction_decision",
        description="Interpret a FullMHT growth-history prediction comparison CSV.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decision = evaluate_growth_history_prediction_decision(
        load_comparison_rows(args.comparison_csv)
    )
    if args.output is not None:
        write_growth_history_prediction_decision(
            decision,
            args.output,
            output_format=str(args.format),
        )
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_growth_history_prediction_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
