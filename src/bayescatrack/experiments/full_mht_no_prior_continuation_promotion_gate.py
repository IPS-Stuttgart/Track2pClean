"""Promotion gate for FullMHT no-prior continuation evidence.

The no-prior continuation likelihood is only paper-facing if it improves or
stabilizes the benchmark without turning into broad non-prior linking.  This
helper combines the frozen comparison CSV with the label-free exposure audit and
keeps the decision rule mechanical.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayescatrack.experiments.full_mht_history_dynamics_decision import (
    load_comparison_rows,
)
from bayescatrack.experiments.full_mht_history_dynamics_promotion_gate import (
    HistoryDynamicsPromotionConfig,
    evaluate_exposure_gate,
    load_exposure_rows,
)
from bayescatrack.experiments.full_mht_no_prior_continuation_decision import (
    evaluate_no_prior_continuation_decision,
)

NO_PRIOR_EXPOSURE_COLUMNS = (
    "max_no_prior_continuation_scored_edges_per_subject",
    "max_no_prior_continuation_positive_edges_per_subject",
    "history_no_prior_continuation_scored_edges",
    "history_no_prior_continuation_positive_edges",
)


@dataclass(frozen=True)
class NoPriorContinuationPromotionConfig:
    """Predeclared exposure limits for no-prior continuation promotion."""

    max_selected_non_prior_edges_per_subject: int = 3
    max_total_non_prior_edges: int = 10
    max_switched_prior_successors: int = 0
    max_no_prior_successor_continuations: int = 10
    max_no_prior_scored_edges_per_subject: int = 6
    max_no_prior_positive_edges_per_subject: int = 3
    max_total_no_prior_positive_edges: int = 10


def evaluate_no_prior_continuation_promotion(
    comparison_rows: Sequence[Mapping[str, Any]],
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: NoPriorContinuationPromotionConfig | None = None,
) -> dict[str, Any]:
    """Combine benchmark and exposure evidence for no-prior continuation."""

    cfg = config or NoPriorContinuationPromotionConfig()
    benchmark = evaluate_no_prior_continuation_decision(comparison_rows)
    exposure = evaluate_no_prior_continuation_exposure(exposure_rows, config=cfg)
    benchmark_result = str(benchmark.get("no_prior_continuation_result", "incomplete"))
    exposure_result = str(exposure.get("exposure_result", "incomplete"))
    if benchmark.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun no-prior continuation benchmark probe"
    elif exposure.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun label-free no-prior continuation exposure audit"
    elif (
        benchmark_result == "no_prior_continuation_stable_gain"
        and exposure_result == "bounded_exposure"
    ):
        status = "promotable_after_review"
        recommendation = "promote only after recording exact output directories"
    elif benchmark_result == "no_prior_continuation_stable_gain":
        status = "not_promotable_broad_exposure"
        recommendation = "keep exploratory; no-prior continuation exposure is too broad"
    elif exposure_result == "bounded_exposure":
        status = "not_promotable_no_stable_gain"
        recommendation = "keep exploratory; benchmark gain is absent or knife-edge"
    else:
        status = "not_promotable"
        recommendation = "keep exploratory; benchmark and exposure gates failed"
    return {
        "status": status,
        "recommendation": recommendation,
        "benchmark_result": benchmark_result,
        "exposure_result": exposure_result,
        "benchmark": benchmark,
        "exposure": exposure,
    }


def evaluate_no_prior_continuation_exposure(
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: NoPriorContinuationPromotionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate whether no-prior continuation exposure remains bounded."""

    cfg = config or NoPriorContinuationPromotionConfig()
    base = evaluate_exposure_gate(
        exposure_rows,
        config=HistoryDynamicsPromotionConfig(
            max_selected_non_prior_edges_per_subject=int(
                cfg.max_selected_non_prior_edges_per_subject
            ),
            max_total_non_prior_edges=int(cfg.max_total_non_prior_edges),
            max_switched_prior_successors=int(cfg.max_switched_prior_successors),
            max_no_prior_successor_continuations=int(
                cfg.max_no_prior_successor_continuations
            ),
        ),
    )
    if base.get("status") != "complete":
        return base
    all_row = _all_row(exposure_rows)
    if all_row is None:
        return {
            "status": "incomplete",
            "exposure_result": "missing_all_row",
            "recommendation": "rerun exposure audit with aggregate ALL row",
        }
    missing_columns = [key for key in NO_PRIOR_EXPOSURE_COLUMNS if key not in all_row]
    if missing_columns:
        return {
            "status": "incomplete",
            "exposure_result": "missing_no_prior_exposure_columns",
            "missing_columns": missing_columns,
            "recommendation": "rerun exposure audit with no-prior continuation scoring enabled",
        }

    failures = list(base.get("failed_limits", ()))
    scored_per_subject = _int_metric(
        all_row,
        "max_no_prior_continuation_scored_edges_per_subject",
    )
    positive_per_subject = _int_metric(
        all_row,
        "max_no_prior_continuation_positive_edges_per_subject",
    )
    total_positive = _int_metric(
        all_row,
        "history_no_prior_continuation_positive_edges",
    )
    total_scored = _int_metric(
        all_row,
        "history_no_prior_continuation_scored_edges",
    )
    if scored_per_subject > int(cfg.max_no_prior_scored_edges_per_subject):
        failures.append("max_no_prior_continuation_scored_edges_per_subject")
    if positive_per_subject > int(cfg.max_no_prior_positive_edges_per_subject):
        failures.append("max_no_prior_continuation_positive_edges_per_subject")
    if total_positive > int(cfg.max_total_no_prior_positive_edges):
        failures.append("history_no_prior_continuation_positive_edges")

    output = dict(base)
    output.update(
        {
            "exposure_result": "bounded_exposure" if not failures else "broad_exposure",
            "failed_limits": failures,
            "history_no_prior_continuation_scored_edges": int(total_scored),
            "history_no_prior_continuation_positive_edges": int(total_positive),
            "max_no_prior_continuation_scored_edges_per_subject": int(
                scored_per_subject
            ),
            "max_no_prior_continuation_positive_edges_per_subject": int(
                positive_per_subject
            ),
            "limit_max_no_prior_continuation_scored_edges_per_subject": int(
                cfg.max_no_prior_scored_edges_per_subject
            ),
            "limit_max_no_prior_continuation_positive_edges_per_subject": int(
                cfg.max_no_prior_positive_edges_per_subject
            ),
            "limit_history_no_prior_continuation_positive_edges": int(
                cfg.max_total_no_prior_positive_edges
            ),
        }
    )
    return output


def format_no_prior_continuation_promotion_markdown(
    decision: Mapping[str, Any],
) -> str:
    """Format a compact no-prior continuation promotion-gate note."""

    exposure = dict(decision.get("exposure", {}))
    lines = [
        "# FullMHT No-Prior Continuation Promotion Gate",
        "",
        f"Status: `{decision.get('status', '')}`",
        f"Benchmark result: `{decision.get('benchmark_result', '')}`",
        f"Exposure result: `{decision.get('exposure_result', '')}`",
        f"Recommendation: {decision.get('recommendation', '')}",
        "",
        "| exposure metric | value | limit |",
        "| --- | ---: | ---: |",
    ]
    for metric, limit_key in (
        (
            "max_selected_non_prior_edges_per_subject",
            "limit_max_selected_non_prior_edges_per_subject",
        ),
        ("history_selected_non_prior_edges", "limit_history_selected_non_prior_edges"),
        (
            "history_no_prior_successor_continuations",
            "limit_history_no_prior_successor_continuations",
        ),
        (
            "max_no_prior_continuation_scored_edges_per_subject",
            "limit_max_no_prior_continuation_scored_edges_per_subject",
        ),
        (
            "max_no_prior_continuation_positive_edges_per_subject",
            "limit_max_no_prior_continuation_positive_edges_per_subject",
        ),
        (
            "history_no_prior_continuation_positive_edges",
            "limit_history_no_prior_continuation_positive_edges",
        ),
    ):
        lines.append(
            "| {metric} | {value} | {limit} |".format(
                metric=metric,
                value=exposure.get(metric, ""),
                limit=exposure.get(limit_key, ""),
            )
        )
    failed = ", ".join(str(item) for item in exposure.get("failed_limits", ()))
    lines.extend(["", f"Failed exposure limits: {failed or 'none'}"])
    if exposure.get("missing_columns"):
        missing = ", ".join(str(item) for item in exposure.get("missing_columns", ()))
        lines.append(f"Missing exposure columns: {missing}")
    return "\n".join(lines)


def write_no_prior_continuation_promotion(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write the promotion gate as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(
        format_no_prior_continuation_promotion_markdown(decision) + "\n",
        encoding="utf-8",
    )


def _all_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("subject", "")) == "ALL":
            return row
    return None


def _int_metric(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(float(row.get(key, 0)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Exposure metric {key!r} is not numeric: {row.get(key)!r}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_no_prior_continuation_promotion_gate",
        description="Combine FullMHT no-prior continuation benchmark and exposure gates.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("exposure_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--max-non-prior-per-subject", type=int, default=3)
    parser.add_argument("--max-total-non-prior", type=int, default=10)
    parser.add_argument("--max-switches", type=int, default=0)
    parser.add_argument("--max-no-prior-continuations", type=int, default=10)
    parser.add_argument("--max-no-prior-scored-per-subject", type=int, default=6)
    parser.add_argument("--max-no-prior-positive-per-subject", type=int, default=3)
    parser.add_argument("--max-total-no-prior-positive", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = NoPriorContinuationPromotionConfig(
        max_selected_non_prior_edges_per_subject=max(
            0,
            int(args.max_non_prior_per_subject),
        ),
        max_total_non_prior_edges=max(0, int(args.max_total_non_prior)),
        max_switched_prior_successors=max(0, int(args.max_switches)),
        max_no_prior_successor_continuations=max(
            0,
            int(args.max_no_prior_continuations),
        ),
        max_no_prior_scored_edges_per_subject=max(
            0,
            int(args.max_no_prior_scored_per_subject),
        ),
        max_no_prior_positive_edges_per_subject=max(
            0,
            int(args.max_no_prior_positive_per_subject),
        ),
        max_total_no_prior_positive_edges=max(
            0,
            int(args.max_total_no_prior_positive),
        ),
    )
    decision = evaluate_no_prior_continuation_promotion(
        load_comparison_rows(args.comparison_csv),
        load_exposure_rows(args.exposure_csv),
        config=cfg,
    )
    if args.output is not None:
        write_no_prior_continuation_promotion(
            decision,
            args.output,
            output_format=str(args.format),
        )
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_no_prior_continuation_promotion_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
