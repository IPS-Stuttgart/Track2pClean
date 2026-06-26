"""Decision helper for the FullMHT identity-history candidate manifest.

The identity-history candidate combines calibrated association likelihood,
calibrated prior-edge survival, no-prior continuation likelihood, and scan-time
growth-history prediction.  This helper keeps the interpretation narrow: the row
is interesting only if the full beam beats its matching greedy beam-width-1
local-choice ablation on complete-track F1 without pairwise-F1 loss.
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
class IdentityHistoryDecisionConfig:
    """Approach names and tolerance for the identity-history candidate."""

    track2p: str = "Track2p"
    prior: str = "FullMHTPrior2"
    prior_survival: str = "FullMHTPriorSurvival"
    no_prior_continuation: str = "FullMHTNoPriorContinuation100"
    identity_history: str = "FullMHTIdentityHistory"
    greedy_identity_history: str = "FullMHTGreedyIdentityHistory"
    tolerance: float = 1e-12


def load_comparison_rows(path: Path) -> list[dict[str, str]]:
    """Load aggregate comparison rows keyed by approach."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No comparison rows found in {path}")
    if "approach" not in rows[0]:
        raise ValueError(f"Comparison CSV {path} is missing an 'approach' column")
    return rows


def evaluate_identity_history_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: IdentityHistoryDecisionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the FullMHT identity-history candidate decision gates."""

    cfg = config or IdentityHistoryDecisionConfig()
    by_approach = _rows_by_approach(rows)
    required = (
        cfg.track2p,
        cfg.prior,
        cfg.prior_survival,
        cfg.no_prior_continuation,
        cfg.identity_history,
        cfg.greedy_identity_history,
    )
    missing = [name for name in required if name not in by_approach]
    if missing:
        return {
            "status": "incomplete",
            "missing_approaches": missing,
            "recommendation": "rerun identity-history manifest with all frozen rows",
        }

    identity_vs_greedy = _delta_block(
        by_approach[cfg.identity_history],
        by_approach[cfg.greedy_identity_history],
        prefix="identity_minus_greedy",
    )
    identity_vs_prior = _delta_block(
        by_approach[cfg.identity_history],
        by_approach[cfg.prior],
        prefix="identity_minus_prior",
    )
    identity_vs_track2p = _delta_block(
        by_approach[cfg.identity_history],
        by_approach[cfg.track2p],
        prefix="identity_minus_track2p",
    )
    identity_vs_survival = _delta_block(
        by_approach[cfg.identity_history],
        by_approach[cfg.prior_survival],
        prefix="identity_minus_prior_survival",
    )
    identity_vs_no_prior = _delta_block(
        by_approach[cfg.identity_history],
        by_approach[cfg.no_prior_continuation],
        prefix="identity_minus_no_prior_continuation",
    )

    history_result = _history_result(
        identity_vs_greedy,
        prefix="identity_minus_greedy",
        tolerance=float(cfg.tolerance),
    )
    prior_result = _control_result(
        identity_vs_prior,
        prefix="identity_minus_prior",
        tolerance=float(cfg.tolerance),
        improvement_name="identity_improves_prior",
        tie_name="identity_ties_prior",
        below_name="identity_below_prior",
    )
    track2p_result = _control_result(
        identity_vs_track2p,
        prefix="identity_minus_track2p",
        tolerance=float(cfg.tolerance),
        improvement_name="identity_improves_track2p",
        tie_name="identity_ties_track2p",
        below_name="identity_below_track2p",
    )
    layer_result = _layer_result(
        identity_vs_survival,
        identity_vs_no_prior,
        tolerance=float(cfg.tolerance),
    )
    recommendation = _recommendation(
        history_result,
        prior_result,
        track2p_result,
        layer_result,
    )
    status = "complete"
    return {
        "status": status,
        "rows": list(required),
        "mht_candidate": cfg.identity_history,
        "local_choice_baseline": cfg.greedy_identity_history,
        "mht_vs_local_result": history_result,
        "history_search_result": history_result,
        "prior_control_result": prior_result,
        "track2p_control_result": track2p_result,
        "layer_combination_result": layer_result,
        "recommendation": recommendation,
        **_mht_vs_local_aliases(identity_vs_greedy),
        **identity_vs_greedy,
        **identity_vs_prior,
        **identity_vs_track2p,
        **identity_vs_survival,
        **identity_vs_no_prior,
    }


def format_decision_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact Markdown decision note."""

    if decision.get("status") != "complete":
        missing = ", ".join(str(item) for item in decision.get("missing_approaches", ()))
        return "\n".join(
            [
                "# FullMHT Identity-History Decision",
                "",
                "Status: incomplete",
                f"Missing approaches: {missing or 'none reported'}",
                f"Recommendation: {decision.get('recommendation', '')}",
            ]
        )

    lines = [
        "# FullMHT Identity-History Decision",
        "",
        f"MHT-vs-local result: `{decision.get('mht_vs_local_result', decision['history_search_result'])}`",
        f"Local-choice baseline: `{decision.get('local_choice_baseline', '')}`",
        f"History-search result: `{decision['history_search_result']}`",
        f"Prior-control result: `{decision['prior_control_result']}`",
        f"Track2p-control result: `{decision['track2p_control_result']}`",
        f"Layer-combination result: `{decision['layer_combination_result']}`",
        f"Recommendation: {decision['recommendation']}",
        "",
        "| comparison | pairwise F1 micro delta | complete-track F1 micro delta |",
        "| --- | ---: | ---: |",
    ]
    for label, prefix in (
        ("MHT minus local greedy", "mht_minus_local"),
        ("identity minus prior", "identity_minus_prior"),
        ("identity minus Track2p", "identity_minus_track2p"),
        ("identity minus prior survival", "identity_minus_prior_survival"),
        ("identity minus no-prior continuation", "identity_minus_no_prior_continuation"),
    ):
        lines.append(
            "| {label} | {pairwise:.6g} | {complete:.6g} |".format(
                label=label,
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


def _mht_vs_local_aliases(identity_vs_greedy: Mapping[str, float]) -> dict[str, float]:
    return {
        f"mht_minus_local_{metric}": float(
            identity_vs_greedy[f"identity_minus_greedy_{metric}"]
        )
        for metric in REPORT_METRICS
    }


def _metric(row: Mapping[str, Any], metric: MetricName) -> float:
    try:
        return float(row[metric])
    except KeyError as exc:
        raise ValueError(f"Comparison row is missing metric column {metric!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Comparison metric {metric!r} is not numeric: {row.get(metric)!r}") from exc


def _history_result(
    deltas: Mapping[str, float],
    *,
    prefix: str,
    tolerance: float,
) -> str:
    pairwise = float(deltas[f"{prefix}_pairwise_f1_micro"])
    complete = float(deltas[f"{prefix}_complete_track_f1_micro"])
    if pairwise < -tolerance or complete < -tolerance:
        return "identity_regression_vs_greedy"
    if complete > tolerance:
        return "identity_complete_history_advantage"
    if pairwise > tolerance:
        return "identity_pairwise_only_advantage"
    return "identity_ties_greedy"


def _control_result(
    deltas: Mapping[str, float],
    *,
    prefix: str,
    tolerance: float,
    improvement_name: str,
    tie_name: str,
    below_name: str,
) -> str:
    values = [float(deltas[f"{prefix}_{metric}"]) for metric in KEY_METRICS]
    if all(delta >= -tolerance for delta in values) and any(delta > tolerance for delta in values):
        return improvement_name
    if all(abs(delta) <= tolerance for delta in values):
        return tie_name
    return below_name


def _layer_result(
    identity_vs_survival: Mapping[str, float],
    identity_vs_no_prior: Mapping[str, float],
    *,
    tolerance: float,
) -> str:
    survival = _control_result(
        identity_vs_survival,
        prefix="identity_minus_prior_survival",
        tolerance=tolerance,
        improvement_name="improves_prior_survival",
        tie_name="ties_prior_survival",
        below_name="below_prior_survival",
    )
    no_prior = _control_result(
        identity_vs_no_prior,
        prefix="identity_minus_no_prior_continuation",
        tolerance=tolerance,
        improvement_name="improves_no_prior_continuation",
        tie_name="ties_no_prior_continuation",
        below_name="below_no_prior_continuation",
    )
    if survival.startswith("below") or no_prior.startswith("below"):
        return "combined_layer_regression"
    if survival.startswith("improves") or no_prior.startswith("improves"):
        return "combined_layer_gain"
    return "combined_layer_tie"


def _recommendation(
    history_result: str,
    prior_result: str,
    track2p_result: str,
    layer_result: str,
) -> str:
    if history_result == "identity_regression_vs_greedy":
        return "do not promote; identity-history beam regresses against matching greedy ablation"
    if history_result == "identity_ties_greedy":
        return "keep exploratory; identity-history beam does not beat greedy history search"
    if history_result == "identity_pairwise_only_advantage":
        return "keep exploratory; identity-history gain is not complete-track advantage"
    if prior_result == "identity_below_prior" or track2p_result == "identity_below_track2p":
        return "keep exploratory; identity-history row loses to a required control"
    if layer_result == "combined_layer_regression":
        return "keep exploratory; combined history model falls below a component layer"
    return "promote only after no-GT, exposure, and sensitivity gates pass"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_identity_history_decision",
        description="Summarize FullMHT identity-history candidate comparison rows.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decision = evaluate_identity_history_decision(load_comparison_rows(args.comparison_csv))
    if args.output is not None:
        write_decision(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_decision_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
