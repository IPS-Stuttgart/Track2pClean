"""Promotion gate for FullMHT growth-history prediction evidence.

This helper combines two frozen artifacts:

* the manual-GT benchmark comparison for the growth-history prediction probe;
* the label-free exposure audit that records how broadly the scan-time
  growth-history penalty fires.

A growth-history prediction row should not be promoted from metrics alone.  It
must show a stable benchmark gain and bounded exposure; otherwise it remains an
exploratory method layer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayescatrack.experiments.full_mht_growth_history_prediction_decision import (
    evaluate_growth_history_prediction_decision,
)
from bayescatrack.experiments.full_mht_history_dynamics_decision import (
    load_comparison_rows,
)


@dataclass(frozen=True)
class GrowthHistoryPredictionPromotionConfig:
    """Predeclared exposure limits for growth-history prediction promotion."""

    max_selected_non_prior_edges_per_subject: int = 3
    max_total_non_prior_edges: int = 10
    max_growth_prediction_penalized_edges_per_subject: int = 3
    max_total_growth_prediction_penalized_edges: int = 20
    max_growth_prediction_weighted_penalty: float = 8.0
    max_growth_prediction_weighted_penalty_per_subject: float = 4.0
    require_growth_prediction_evaluated: bool = True


def load_exposure_rows(path: Path) -> list[dict[str, str]]:
    """Load label-free growth-history exposure-audit rows."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No exposure rows found in {path}")
    if "subject" not in rows[0]:
        raise ValueError(f"Exposure CSV {path} is missing a 'subject' column")
    return rows


def evaluate_growth_history_prediction_promotion(
    comparison_rows: Sequence[Mapping[str, Any]],
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: GrowthHistoryPredictionPromotionConfig | None = None,
) -> dict[str, Any]:
    """Combine benchmark sensitivity and label-free exposure evidence."""

    cfg = config or GrowthHistoryPredictionPromotionConfig()
    benchmark = evaluate_growth_history_prediction_decision(comparison_rows)
    exposure = evaluate_growth_history_prediction_exposure_gate(
        exposure_rows,
        config=cfg,
    )
    benchmark_result = str(
        benchmark.get("growth_history_prediction_result", "incomplete")
    )
    exposure_result = str(exposure.get("exposure_result", "incomplete"))
    if benchmark.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun growth-history prediction benchmark probe"
    elif exposure.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun label-free growth-history exposure audit"
    elif benchmark_result == "history_dynamics_stable_gain" and exposure_result == "bounded_exposure":
        status = "promotable_after_review"
        recommendation = "promote candidate after recording exact output directories"
    elif benchmark_result == "history_dynamics_stable_gain":
        status = "not_promotable_broad_exposure"
        recommendation = "keep exploratory; growth-history exposure is too broad"
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


def evaluate_growth_history_prediction_exposure_gate(
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: GrowthHistoryPredictionPromotionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate whether growth-history prediction exposure remains bounded."""

    cfg = config or GrowthHistoryPredictionPromotionConfig()
    all_row = _all_row(exposure_rows)
    if all_row is None:
        return {
            "status": "incomplete",
            "exposure_result": "missing_all_row",
            "recommendation": "rerun exposure audit with aggregate ALL row",
        }

    max_non_prior = _int_metric(all_row, "max_selected_non_prior_edges_per_subject")
    total_non_prior = _int_metric(all_row, "history_selected_non_prior_edges")
    evaluated = _int_metric(all_row, "history_growth_prediction_evaluated_edges")
    total_penalized = _int_metric(all_row, "history_growth_prediction_penalized_edges")
    max_penalized = _int_metric(
        all_row,
        "max_growth_prediction_penalized_edges_per_subject",
    )
    weighted = _float_metric(all_row, "history_growth_prediction_weighted_penalty")
    max_weighted = _float_metric(
        all_row,
        "max_growth_prediction_weighted_penalty_per_subject",
    )
    if bool(cfg.require_growth_prediction_evaluated) and evaluated <= 0:
        return {
            "status": "incomplete",
            "exposure_result": "growth_prediction_not_evaluated",
            "recommendation": "rerun exposure audit with growth-history prediction enabled",
            "history_growth_prediction_evaluated_edges": int(evaluated),
        }

    failures: list[str] = []
    if max_non_prior > int(cfg.max_selected_non_prior_edges_per_subject):
        failures.append("max_selected_non_prior_edges_per_subject")
    if total_non_prior > int(cfg.max_total_non_prior_edges):
        failures.append("history_selected_non_prior_edges")
    if max_penalized > int(cfg.max_growth_prediction_penalized_edges_per_subject):
        failures.append("max_growth_prediction_penalized_edges_per_subject")
    if total_penalized > int(cfg.max_total_growth_prediction_penalized_edges):
        failures.append("history_growth_prediction_penalized_edges")
    if weighted > float(cfg.max_growth_prediction_weighted_penalty):
        failures.append("history_growth_prediction_weighted_penalty")
    if max_weighted > float(cfg.max_growth_prediction_weighted_penalty_per_subject):
        failures.append("max_growth_prediction_weighted_penalty_per_subject")

    return {
        "status": "complete",
        "exposure_result": "bounded_exposure" if not failures else "broad_exposure",
        "failed_limits": failures,
        "max_selected_non_prior_edges_per_subject": int(max_non_prior),
        "history_selected_non_prior_edges": int(total_non_prior),
        "history_growth_prediction_evaluated_edges": int(evaluated),
        "history_growth_prediction_penalized_edges": int(total_penalized),
        "max_growth_prediction_penalized_edges_per_subject": int(max_penalized),
        "history_growth_prediction_weighted_penalty": float(weighted),
        "max_growth_prediction_weighted_penalty_per_subject": float(max_weighted),
        "limit_max_selected_non_prior_edges_per_subject": int(
            cfg.max_selected_non_prior_edges_per_subject
        ),
        "limit_history_selected_non_prior_edges": int(cfg.max_total_non_prior_edges),
        "limit_max_growth_prediction_penalized_edges_per_subject": int(
            cfg.max_growth_prediction_penalized_edges_per_subject
        ),
        "limit_history_growth_prediction_penalized_edges": int(
            cfg.max_total_growth_prediction_penalized_edges
        ),
        "limit_history_growth_prediction_weighted_penalty": float(
            cfg.max_growth_prediction_weighted_penalty
        ),
        "limit_max_growth_prediction_weighted_penalty_per_subject": float(
            cfg.max_growth_prediction_weighted_penalty_per_subject
        ),
    }


def format_promotion_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact growth-history promotion-gate Markdown note."""

    exposure = dict(decision.get("exposure", {}))
    lines = [
        "# FullMHT Growth-History Prediction Promotion Gate",
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
            "history_growth_prediction_penalized_edges",
            "limit_history_growth_prediction_penalized_edges",
        ),
        (
            "max_growth_prediction_penalized_edges_per_subject",
            "limit_max_growth_prediction_penalized_edges_per_subject",
        ),
        (
            "history_growth_prediction_weighted_penalty",
            "limit_history_growth_prediction_weighted_penalty",
        ),
        (
            "max_growth_prediction_weighted_penalty_per_subject",
            "limit_max_growth_prediction_weighted_penalty_per_subject",
        ),
    ):
        lines.append(
            "| {metric} | {value} | {limit} |".format(
                metric=metric,
                value=exposure.get(metric, ""),
                limit=exposure.get(limit_key, ""),
            )
        )
    lines.extend(
        [
            "",
            "Growth-history evaluated edges: {value}".format(
                value=exposure.get("history_growth_prediction_evaluated_edges", "")
            ),
            "Failed exposure limits: {failed}".format(
                failed=", ".join(
                    str(item) for item in exposure.get("failed_limits", ())
                )
                or "none"
            ),
        ]
    )
    return "\n".join(lines)


def write_promotion(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write the promotion gate as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(format_promotion_markdown(decision) + "\n", encoding="utf-8")


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


def _float_metric(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Exposure metric {key!r} is not numeric: {row.get(key)!r}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "bayescatrack.experiments.full_mht_growth_history_prediction_promotion_gate"
        ),
        description="Combine FullMHT growth-history benchmark and exposure gates.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("exposure_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--max-non-prior-per-subject", type=int, default=3)
    parser.add_argument("--max-total-non-prior", type=int, default=10)
    parser.add_argument("--max-growth-penalized-per-subject", type=int, default=3)
    parser.add_argument("--max-total-growth-penalized", type=int, default=20)
    parser.add_argument("--max-growth-weighted-penalty", type=float, default=8.0)
    parser.add_argument("--max-growth-weighted-penalty-per-subject", type=float, default=4.0)
    parser.add_argument(
        "--require-growth-prediction-evaluated",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = GrowthHistoryPredictionPromotionConfig(
        max_selected_non_prior_edges_per_subject=max(
            0,
            int(args.max_non_prior_per_subject),
        ),
        max_total_non_prior_edges=max(0, int(args.max_total_non_prior)),
        max_growth_prediction_penalized_edges_per_subject=max(
            0,
            int(args.max_growth_penalized_per_subject),
        ),
        max_total_growth_prediction_penalized_edges=max(
            0,
            int(args.max_total_growth_penalized),
        ),
        max_growth_prediction_weighted_penalty=max(
            0.0,
            float(args.max_growth_weighted_penalty),
        ),
        max_growth_prediction_weighted_penalty_per_subject=max(
            0.0,
            float(args.max_growth_weighted_penalty_per_subject),
        ),
        require_growth_prediction_evaluated=bool(args.require_growth_prediction_evaluated),
    )
    decision = evaluate_growth_history_prediction_promotion(
        load_comparison_rows(args.comparison_csv),
        load_exposure_rows(args.exposure_csv),
        config=cfg,
    )
    if args.output is not None:
        write_promotion(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_promotion_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
