"""Promotion gate for FullMHT history-dynamics evidence.

This helper combines two frozen artifacts:

* the manual-GT benchmark comparison for the history-dynamics sensitivity probe;
* the label-free exposure audit across Track2p-style subjects.

A FullMHT method layer should not be promoted from metrics alone.  It must show a
stable benchmark gain and bounded exposure, otherwise it remains exploratory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayescatrack.experiments.full_mht_history_dynamics_decision import (
    evaluate_history_dynamics_decision,
    load_comparison_rows,
)


@dataclass(frozen=True)
class HistoryDynamicsPromotionConfig:
    """Predeclared exposure limits for a history-dynamics promotion gate."""

    max_selected_non_prior_edges_per_subject: int = 3
    max_total_non_prior_edges: int = 10
    max_switched_prior_successors: int = 0
    max_no_prior_successor_continuations: int = 10
    max_gap_reactivated_tracks: int | None = None


def load_exposure_rows(path: Path) -> list[dict[str, str]]:
    """Load label-free exposure-audit rows."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No exposure rows found in {path}")
    if "subject" not in rows[0]:
        raise ValueError(f"Exposure CSV {path} is missing a 'subject' column")
    return rows


def evaluate_history_dynamics_promotion(
    comparison_rows: Sequence[Mapping[str, Any]],
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: HistoryDynamicsPromotionConfig | None = None,
) -> dict[str, Any]:
    """Combine benchmark sensitivity and label-free exposure evidence."""

    cfg = config or HistoryDynamicsPromotionConfig()
    benchmark = evaluate_history_dynamics_decision(comparison_rows)
    exposure = evaluate_exposure_gate(exposure_rows, config=cfg)
    benchmark_result = str(benchmark.get("history_dynamics_result", "incomplete"))
    exposure_result = str(exposure.get("exposure_result", "incomplete"))
    if benchmark.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun history-dynamics benchmark probe"
    elif exposure.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun label-free FullMHT exposure audit"
    elif benchmark_result == "history_dynamics_stable_gain" and exposure_result == "bounded_exposure":
        status = "promotable_after_review"
        recommendation = "promote candidate after recording exact output directories"
    elif benchmark_result == "history_dynamics_stable_gain":
        status = "not_promotable_broad_exposure"
        recommendation = "keep exploratory; label-free exposure is too broad"
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


def evaluate_exposure_gate(
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: HistoryDynamicsPromotionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate whether label-free exposure remains bounded."""

    cfg = config or HistoryDynamicsPromotionConfig()
    all_row = _all_row(exposure_rows)
    if all_row is None:
        return {
            "status": "incomplete",
            "exposure_result": "missing_all_row",
            "recommendation": "rerun exposure audit with aggregate ALL row",
        }

    max_non_prior = _int_metric(all_row, "max_selected_non_prior_edges_per_subject")
    total_non_prior = _int_metric(all_row, "history_selected_non_prior_edges")
    switches = _int_metric(all_row, "history_switched_prior_successors")
    no_prior = _int_metric(all_row, "history_no_prior_successor_continuations")
    gap_reactivations = _int_metric(all_row, "history_gap_reactivated_tracks")
    failures: list[str] = []
    if max_non_prior > int(cfg.max_selected_non_prior_edges_per_subject):
        failures.append("max_selected_non_prior_edges_per_subject")
    if total_non_prior > int(cfg.max_total_non_prior_edges):
        failures.append("history_selected_non_prior_edges")
    if switches > int(cfg.max_switched_prior_successors):
        failures.append("history_switched_prior_successors")
    if no_prior > int(cfg.max_no_prior_successor_continuations):
        failures.append("history_no_prior_successor_continuations")
    if cfg.max_gap_reactivated_tracks is not None and gap_reactivations > int(
        cfg.max_gap_reactivated_tracks
    ):
        failures.append("history_gap_reactivated_tracks")

    return {
        "status": "complete",
        "exposure_result": "bounded_exposure" if not failures else "broad_exposure",
        "failed_limits": failures,
        "max_selected_non_prior_edges_per_subject": int(max_non_prior),
        "history_selected_non_prior_edges": int(total_non_prior),
        "history_switched_prior_successors": int(switches),
        "history_no_prior_successor_continuations": int(no_prior),
        "history_gap_reactivated_tracks": int(gap_reactivations),
        "limit_max_selected_non_prior_edges_per_subject": int(
            cfg.max_selected_non_prior_edges_per_subject
        ),
        "limit_history_selected_non_prior_edges": int(cfg.max_total_non_prior_edges),
        "limit_history_switched_prior_successors": int(cfg.max_switched_prior_successors),
        "limit_history_no_prior_successor_continuations": int(
            cfg.max_no_prior_successor_continuations
        ),
    }


def format_promotion_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact promotion-gate Markdown note."""

    exposure = dict(decision.get("exposure", {}))
    lines = [
        "# FullMHT History Dynamics Promotion Gate",
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
            "history_switched_prior_successors",
            "limit_history_switched_prior_successors",
        ),
        (
            "history_no_prior_successor_continuations",
            "limit_history_no_prior_successor_continuations",
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_history_dynamics_promotion_gate",
        description="Combine FullMHT history-dynamics benchmark and exposure gates.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("exposure_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--max-non-prior-per-subject", type=int, default=3)
    parser.add_argument("--max-total-non-prior", type=int, default=10)
    parser.add_argument("--max-switches", type=int, default=0)
    parser.add_argument("--max-no-prior-continuations", type=int, default=10)
    parser.add_argument("--max-gap-reactivations", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = HistoryDynamicsPromotionConfig(
        max_selected_non_prior_edges_per_subject=max(
            0,
            int(args.max_non_prior_per_subject),
        ),
        max_total_non_prior=max(0, int(args.max_total_non_prior)),
        max_switched_prior_successors=max(0, int(args.max_switches)),
        max_no_prior_successor_continuations=max(
            0,
            int(args.max_no_prior_continuations),
        ),
        max_gap_reactivated_tracks=(
            None
            if args.max_gap_reactivations is None
            else max(0, int(args.max_gap_reactivations))
        ),
    )
    decision = evaluate_history_dynamics_promotion(
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
