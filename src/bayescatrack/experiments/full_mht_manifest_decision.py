"""Decision helper for FullMHT manifest comparison outputs.

The FullMHT method story depends on three separate questions:

* does a full beam beat the matching greedy beam-width-1 ablation on complete-track
  F1 with no pairwise-F1 loss, showing a real identity-history search advantage?
* does the fixed prior-veto row remain a positive FullMHT-owned scan-assignment
  result when compared with its own greedy ablation?
* does the calibrated prior-survival row match or improve the fixed prior-veto
  hazard, making the row less hand-gated?

This module reads the aggregate comparison CSV produced by
``bayescatrack benchmark compare`` / benchmark-suite comparisons and emits a small
machine-readable and Markdown decision summary.  It is intentionally separate
from the benchmark runner so the interpretation can be rerun after any manifest
execution without touching the result files.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MetricName = Literal[
    "pairwise_f1_micro",
    "complete_track_f1_micro",
    "pairwise_f1_macro",
    "complete_track_f1_macro",
]

KEY_METRICS: tuple[MetricName, ...] = (
    "pairwise_f1_micro",
    "complete_track_f1_micro",
)
REPORT_METRICS: tuple[MetricName, ...] = (
    "pairwise_f1_micro",
    "complete_track_f1_micro",
    "pairwise_f1_macro",
    "complete_track_f1_macro",
)


@dataclass(frozen=True)
class FullMHTDecisionConfig:
    """Approach names and tolerance for a FullMHT manifest decision."""

    track2p: str = "Track2p"
    beam: str = "FullMHTPrior2"
    greedy: str = "FullMHTGreedyPrior2"
    prior_veto: str = "FullMHTPriorVetoScaled"
    greedy_prior_veto: str = "FullMHTGreedyPriorVetoScaled"
    prior_survival: str = "FullMHTPriorSurvival"
    greedy_prior_survival: str = "FullMHTGreedyPriorSurvival"
    tolerance: float = 1e-12


def load_comparison_rows(path: Path) -> list[dict[str, str]]:
    """Load aggregate comparison rows keyed by ``approach``."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No comparison rows found in {path}")
    if "approach" not in rows[0]:
        raise ValueError(f"Comparison CSV {path} is missing an 'approach' column")
    return rows


def evaluate_full_mht_manifest_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: FullMHTDecisionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the FullMHT manifest promotion gates from comparison rows."""

    cfg = config or FullMHTDecisionConfig()
    by_approach = _rows_by_approach(rows)
    required = (
        cfg.track2p,
        cfg.beam,
        cfg.greedy,
        cfg.prior_veto,
        cfg.greedy_prior_veto,
        cfg.prior_survival,
        cfg.greedy_prior_survival,
    )
    missing = [name for name in required if name not in by_approach]
    if missing:
        return {
            "status": "incomplete",
            "missing_approaches": missing,
            "recommendation": "rerun manifest with all canonical FullMHT rows",
        }

    base_beam_vs_greedy = _delta_block(
        by_approach[cfg.beam], by_approach[cfg.greedy], prefix="base_beam_minus_greedy"
    )
    veto_beam_vs_greedy = _delta_block(
        by_approach[cfg.prior_veto],
        by_approach[cfg.greedy_prior_veto],
        prefix="veto_beam_minus_greedy",
    )
    survival_beam_vs_greedy = _delta_block(
        by_approach[cfg.prior_survival],
        by_approach[cfg.greedy_prior_survival],
        prefix="survival_beam_minus_greedy",
    )
    veto_vs_beam = _delta_block(
        by_approach[cfg.prior_veto], by_approach[cfg.beam], prefix="veto_minus_beam"
    )
    survival_vs_veto = _delta_block(
        by_approach[cfg.prior_survival],
        by_approach[cfg.prior_veto],
        prefix="survival_minus_veto",
    )
    survival_vs_track2p = _delta_block(
        by_approach[cfg.prior_survival],
        by_approach[cfg.track2p],
        prefix="survival_minus_track2p",
    )

    base_history_result = _history_result(
        base_beam_vs_greedy,
        prefix="base_beam_minus_greedy",
        tolerance=float(cfg.tolerance),
    )
    veto_history_result = _history_result(
        veto_beam_vs_greedy,
        prefix="veto_beam_minus_greedy",
        tolerance=float(cfg.tolerance),
    )
    survival_history_result = _history_result(
        survival_beam_vs_greedy,
        prefix="survival_beam_minus_greedy",
        tolerance=float(cfg.tolerance),
    )
    candidate_history_result = _candidate_history_result(
        veto_history_result,
        survival_history_result,
    )
    survival_result = _survival_result(
        survival_vs_veto,
        survival_vs_track2p,
        tolerance=float(cfg.tolerance),
    )
    recommendation = _recommendation(
        candidate_history_result,
        survival_result,
        veto_history_result,
    )
    return {
        "status": "complete",
        "rows": list(required),
        "history_search_result": candidate_history_result,
        "base_history_search_result": base_history_result,
        "prior_veto_history_search_result": veto_history_result,
        "prior_survival_history_search_result": survival_history_result,
        "prior_survival_result": survival_result,
        "recommendation": recommendation,
        **base_beam_vs_greedy,
        **veto_beam_vs_greedy,
        **survival_beam_vs_greedy,
        **veto_vs_beam,
        **survival_vs_veto,
        **survival_vs_track2p,
    }


def format_decision_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact Markdown decision note."""

    if decision.get("status") != "complete":
        missing = ", ".join(str(item) for item in decision.get("missing_approaches", ()))
        return "\n".join(
            [
                "# FullMHT Manifest Decision",
                "",
                "Status: incomplete",
                f"Missing approaches: {missing or 'none reported'}",
                f"Recommendation: {decision.get('recommendation', '')}",
            ]
        )

    lines = [
        "# FullMHT Manifest Decision",
        "",
        f"History-search result: `{decision['history_search_result']}`",
        f"Base history-search result: `{decision['base_history_search_result']}`",
        f"Prior-veto history-search result: `{decision['prior_veto_history_search_result']}`",
        f"Prior-survival history-search result: `{decision['prior_survival_history_search_result']}`",
        f"Prior-survival result: `{decision['prior_survival_result']}`",
        f"Recommendation: {decision['recommendation']}",
        "",
        "| comparison | pairwise F1 micro delta | complete-track F1 micro delta |",
        "| --- | ---: | ---: |",
    ]
    for prefix in (
        "base_beam_minus_greedy",
        "veto_beam_minus_greedy",
        "survival_beam_minus_greedy",
        "veto_minus_beam",
        "survival_minus_veto",
        "survival_minus_track2p",
    ):
        lines.append(
            "| {label} | {pairwise:.6g} | {complete:.6g} |".format(
                label=prefix.replace("_", " "),
                pairwise=float(decision[f"{prefix}_pairwise_f1_micro"]),
                complete=float(decision[f"{prefix}_complete_track_f1_micro"]),
            )
        )
    return "\n".join(lines)


def write_decision(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write a decision artifact as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(format_decision_markdown(decision) + "\n", encoding="utf-8")


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


def _metric(row: Mapping[str, Any], metric: MetricName) -> float:
    try:
        return float(row[metric])
    except KeyError as exc:
        raise ValueError(f"Comparison row is missing metric column {metric!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Comparison metric {metric!r} is not numeric: {row[metric]!r}") from exc


def _history_result(
    deltas: Mapping[str, float],
    *,
    prefix: str,
    tolerance: float,
) -> str:
    pairwise = float(deltas[f"{prefix}_pairwise_f1_micro"])
    complete = float(deltas[f"{prefix}_complete_track_f1_micro"])
    if pairwise < -tolerance or complete < -tolerance:
        return "beam_regression_vs_greedy"
    if complete > tolerance:
        return "beam_complete_history_advantage"
    if pairwise > tolerance:
        return "beam_pairwise_only_advantage"
    return "beam_ties_greedy"


def _candidate_history_result(veto_result: str, survival_result: str) -> str:
    if survival_result == "beam_complete_history_advantage":
        return "prior_survival_complete_history_advantage"
    if veto_result == "beam_complete_history_advantage":
        return "fixed_veto_complete_history_advantage"
    if "beam_regression_vs_greedy" in {veto_result, survival_result}:
        return "candidate_regression_vs_greedy"
    if "beam_pairwise_only_advantage" in {veto_result, survival_result}:
        return "candidate_pairwise_only_advantage"
    return "candidate_ties_greedy"


def _survival_result(
    survival_vs_veto: Mapping[str, float],
    survival_vs_track2p: Mapping[str, float],
    *,
    tolerance: float,
) -> str:
    vs_veto = [
        float(survival_vs_veto[f"survival_minus_veto_{metric}"])
        for metric in KEY_METRICS
    ]
    vs_track2p = [
        float(survival_vs_track2p[f"survival_minus_track2p_{metric}"])
        for metric in KEY_METRICS
    ]
    if all(delta >= -tolerance for delta in vs_veto) and any(
        delta > tolerance for delta in vs_veto
    ):
        return "survival_improves_fixed_veto"
    if all(abs(delta) <= tolerance for delta in vs_veto):
        return "survival_ties_fixed_veto"
    if all(delta >= -tolerance for delta in vs_track2p):
        return "survival_above_track2p_but_below_fixed_veto"
    return "survival_not_promotable"


def _recommendation(
    candidate_history_result: str,
    survival_result: str,
    veto_history_result: str,
) -> str:
    if candidate_history_result == "candidate_regression_vs_greedy":
        return "do not promote FullMHT; investigate candidate beam scoring regression"
    if candidate_history_result == "candidate_ties_greedy":
        return "keep FullMHT exploratory; candidate rows do not beat greedy history search"
    if candidate_history_result == "candidate_pairwise_only_advantage":
        return "keep FullMHT exploratory; candidate beam gain is not complete-history advantage"
    if survival_result in {
        "survival_improves_fixed_veto",
        "survival_ties_fixed_veto",
    } and candidate_history_result == "prior_survival_complete_history_advantage":
        return "promote prior-survival candidate only after no-GT, exposure, and sensitivity gates pass"
    if veto_history_result == "beam_complete_history_advantage":
        return "keep prior-survival exploratory; fixed prior-veto is the current FullMHT history-search row"
    return "keep FullMHT exploratory until a candidate row shows complete-history beam advantage"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_manifest_decision",
        description="Summarize canonical FullMHT manifest comparison rows.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decision = evaluate_full_mht_manifest_decision(load_comparison_rows(args.comparison_csv))
    if args.output is not None:
        write_decision(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_decision_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
