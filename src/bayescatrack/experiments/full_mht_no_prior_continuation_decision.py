"""Decision helper for FullMHT no-prior continuation probe manifests.

The no-prior continuation likelihood asks whether FullMHT can distinguish a real
continuation from a death/miss when Track2p has no proposal successor.  This
helper freezes the metric-side interpretation of the probe: a candidate is
interesting only if it improves complete-track F1 without pairwise-F1 loss across
more than one nearby likelihood weight, while the deliberately permissive
``FullMHTCalibratedNoDeath`` control remains visible in the report.
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
    load_comparison_rows,
)

NO_PRIOR_CONTINUATION_CONFIG = HistoryDynamicsDecisionConfig(
    baseline="FullMHTPrior2",
    candidates=(
        "FullMHTNoPriorContinuation050",
        "FullMHTNoPriorContinuation100",
        "FullMHTNoPriorContinuation150",
    ),
    track2p="Track2p",
)
PERMISSIVE_CONTROL = "FullMHTCalibratedNoDeath"
REPORT_METRICS: tuple[str, ...] = (
    "pairwise_f1_micro",
    "complete_track_f1_micro",
    "pairwise_f1_macro",
    "complete_track_f1_macro",
)


def evaluate_no_prior_continuation_decision(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate whether no-prior continuation likelihood is promotable."""

    by_approach = _rows_by_approach(rows)
    required = (
        NO_PRIOR_CONTINUATION_CONFIG.track2p,
        NO_PRIOR_CONTINUATION_CONFIG.baseline,
        PERMISSIVE_CONTROL,
        *NO_PRIOR_CONTINUATION_CONFIG.candidates,
    )
    missing = [name for name in required if name not in by_approach]
    if missing:
        return {
            "status": "incomplete",
            "missing_approaches": missing,
            "recommendation": "rerun no-prior continuation probe manifest",
        }

    decision = evaluate_history_dynamics_decision(
        rows,
        config=NO_PRIOR_CONTINUATION_CONFIG,
    )
    if decision.get("status") != "complete":
        decision = dict(decision)
        decision["recommendation"] = "rerun no-prior continuation probe manifest"
        return decision

    decision = dict(decision)
    raw_result = str(decision.pop("history_dynamics_result"))
    result = _no_prior_result(raw_result)
    decision["no_prior_continuation_result"] = result
    decision["permissive_control"] = PERMISSIVE_CONTROL
    decision["recommendation"] = _no_prior_recommendation(result)
    decision.update(
        _delta_block(
            by_approach[str(decision["best_candidate"])],
            by_approach[PERMISSIVE_CONTROL],
            prefix="best_minus_permissive_control",
        )
    )
    return decision


def format_no_prior_continuation_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact no-prior continuation decision note."""

    if decision.get("status") != "complete":
        missing = ", ".join(str(item) for item in decision.get("missing_approaches", ()))
        return "\n".join(
            [
                "# FullMHT No-Prior Continuation Decision",
                "",
                "Status: incomplete",
                f"Missing approaches: {missing or 'none reported'}",
                f"Recommendation: {decision.get('recommendation', '')}",
            ]
        )

    lines = [
        "# FullMHT No-Prior Continuation Decision",
        "",
        f"Result: `{decision['no_prior_continuation_result']}`",
        f"Best candidate: `{decision['best_candidate']}`",
        f"Permissive control: `{decision['permissive_control']}`",
        f"Recommendation: {decision['recommendation']}",
        "",
        "| candidate | decision | pairwise F1 micro delta | complete-track F1 micro delta | pairwise F1 macro delta | complete-track F1 macro delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for block in decision.get("candidate_decisions", ()):  # type: ignore[assignment]
        row = dict(block)
        lines.append(
            "| {approach} | {decision} | {pairwise_micro:.6g} | {complete_micro:.6g} | {pairwise_macro:.6g} | {complete_macro:.6g} |".format(
                approach=row["approach"],
                decision=row["decision"],
                pairwise_micro=float(row["delta_pairwise_f1_micro"]),
                complete_micro=float(row["delta_complete_track_f1_micro"]),
                pairwise_macro=float(row["delta_pairwise_f1_macro"]),
                complete_macro=float(row["delta_complete_track_f1_macro"]),
            )
        )
    lines.extend(
        [
            "",
            "Best candidate minus permissive control:",
            "",
            "| metric | delta |",
            "| --- | ---: |",
        ]
    )
    for metric in REPORT_METRICS:
        key = f"best_minus_permissive_control_{metric}"
        lines.append(f"| {metric} | {float(decision[key]):.6g} |")
    return "\n".join(lines)


def write_no_prior_continuation_decision(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write a no-prior continuation decision artifact as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(
        format_no_prior_continuation_markdown(decision) + "\n",
        encoding="utf-8",
    )


def _rows_by_approach(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_approach: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        approach = str(row.get("approach", ""))
        if approach:
            by_approach[approach] = row
    return by_approach


def _delta_block(
    candidate: Mapping[str, Any], reference: Mapping[str, Any], *, prefix: str
) -> dict[str, float]:
    return {
        f"{prefix}_{metric}": _metric(candidate, metric) - _metric(reference, metric)
        for metric in REPORT_METRICS
    }


def _metric(row: Mapping[str, Any], metric: str) -> float:
    try:
        return float(row[metric])
    except KeyError as exc:
        raise ValueError(f"Comparison row is missing metric column {metric!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Comparison metric {metric!r} is not numeric: {row[metric]!r}") from exc


def _no_prior_result(history_result: str) -> str:
    suffix_by_result = {
        "history_dynamics_stable_gain": "stable_gain",
        "history_dynamics_single_weight_gain": "single_weight_gain",
        "history_dynamics_ties_baseline": "ties_baseline",
        "history_dynamics_pairwise_regression": "pairwise_regression",
        "history_dynamics_complete_regression": "complete_regression",
        "history_dynamics_not_promotable": "not_promotable",
    }
    suffix = suffix_by_result.get(history_result, "not_promotable")
    return f"no_prior_continuation_{suffix}"


def _no_prior_recommendation(result: str) -> str:
    if result == "no_prior_continuation_stable_gain":
        return (
            "promote only after exposure/no-GT checks show rare no-prior "
            "continuations; this is a promising birth/death likelihood layer"
        )
    if result == "no_prior_continuation_single_weight_gain":
        return "treat as exploratory; gain appears knife-edge"
    if result == "no_prior_continuation_ties_baseline":
        return "record as likelihood-equivalent to the proposal-prior control"
    if result == "no_prior_continuation_pairwise_regression":
        return "do not promote; no-prior likelihood damages pairwise tracking"
    if result == "no_prior_continuation_complete_regression":
        return "do not promote; no-prior likelihood damages complete-track identity"
    return "do not promote no-prior continuation likelihood from this probe"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_no_prior_continuation_decision",
        description="Interpret a FullMHT no-prior continuation comparison CSV.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decision = evaluate_no_prior_continuation_decision(
        load_comparison_rows(args.comparison_csv)
    )
    if args.output is not None:
        write_no_prior_continuation_decision(
            decision,
            args.output,
            output_format=str(args.format),
        )
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_no_prior_continuation_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
