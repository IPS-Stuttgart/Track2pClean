"""Decision helper for the FullMHT local-neighborhood context probe.

The local context layer penalizes candidate continuations whose displacement is
incoherent with nearby high-confidence anchor matches.  This helper interprets a
small frozen weight sweep against the same FullMHT prior baseline with the local
context term disabled.
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

LOCAL_CONTEXT_CONFIG = HistoryDynamicsDecisionConfig(
    baseline="FullMHTLocalContext000",
    candidates=(
        "FullMHTLocalContext025",
        "FullMHTLocalContext050",
        "FullMHTLocalContext100",
    ),
    track2p="Track2p",
)


def evaluate_local_context_decision(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate whether local-neighborhood context is stable enough to promote."""

    decision = evaluate_history_dynamics_decision(
        rows,
        config=LOCAL_CONTEXT_CONFIG,
    )
    if decision.get("status") == "complete":
        decision = dict(decision)
        decision["local_context_result"] = decision.pop("history_dynamics_result")
        decision["recommendation"] = _local_context_recommendation(
            str(decision["local_context_result"])
        )
    return decision


def format_local_context_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact local-context decision note."""

    if decision.get("status") != "complete":
        missing = ", ".join(str(item) for item in decision.get("missing_approaches", ()))
        return "\n".join(
            [
                "# FullMHT Local-Context Decision",
                "",
                "Status: incomplete",
                f"Missing approaches: {missing or 'none reported'}",
                f"Recommendation: {decision.get('recommendation', '')}",
            ]
        )

    generic = format_decision_markdown(
        {**dict(decision), "history_dynamics_result": decision["local_context_result"]}
    )
    return generic.replace(
        "# FullMHT History Dynamics Decision",
        "# FullMHT Local-Context Decision",
        1,
    )


def write_local_context_decision(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write a local-context decision artifact as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(format_local_context_markdown(decision) + "\n", encoding="utf-8")


def _local_context_recommendation(result: str) -> str:
    if result == "history_dynamics_stable_gain":
        return (
            "promote as a promising label-free neighborhood-coherence component "
            "after exposure/no-GT checks"
        )
    if result == "history_dynamics_single_weight_gain":
        return "treat as exploratory; local-context gain appears knife-edge"
    if result == "history_dynamics_ties_baseline":
        return "record as local-context-equivalent to the no-context control"
    if result == "history_dynamics_pairwise_regression":
        return "do not promote; local context damages pairwise tracking"
    if result == "history_dynamics_complete_regression":
        return "do not promote; local context damages complete-track identity"
    return "rerun or inspect probe outputs before interpreting"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_local_context_decision",
        description="Interpret a FullMHT local-neighborhood context comparison CSV.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decision = evaluate_local_context_decision(load_comparison_rows(args.comparison_csv))
    if args.output is not None:
        write_local_context_decision(
            decision,
            args.output,
            output_format=str(args.format),
        )
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_local_context_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
