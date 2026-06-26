"""Decision helper for identity-history scan-pruning add-on probes.

The scan-history pruning add-on is interesting only if it makes MHT history search
load-bearing on real benchmark rows.  For every frozen weight, the full beam is
compared with a matching greedy beam-width-1 row that uses the same scoring terms.
The add-on must also avoid regression against the central identity-history row.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

MetricName = Literal[
    "pairwise_f1_micro",
    "complete_track_f1_micro",
    "pairwise_f1_macro",
    "complete_track_f1_macro",
]

REPORT_METRICS: tuple[MetricName, ...] = (
    "pairwise_f1_micro",
    "complete_track_f1_micro",
    "pairwise_f1_macro",
    "complete_track_f1_macro",
)
SCAN_PRUNING_PAIRS: tuple[tuple[str, str], ...] = (
    ("IdentityHistoryScanPruning025", "GreedyIdentityHistoryScanPruning025"),
    ("IdentityHistoryScanPruning050", "GreedyIdentityHistoryScanPruning050"),
    ("IdentityHistoryScanPruning100", "GreedyIdentityHistoryScanPruning100"),
)


def load_comparison_rows(path: Path) -> list[dict[str, str]]:
    """Load aggregate comparison rows keyed by approach."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No comparison rows found in {path}")
    if "approach" not in rows[0]:
        raise ValueError(f"Comparison CSV {path} is missing an 'approach' column")
    return rows


def evaluate_identity_history_scan_pruning_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str = "FullMHTIdentityHistory",
    pairs: Sequence[tuple[str, str]] = SCAN_PRUNING_PAIRS,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate whether scan-history pruning is promotable as an add-on."""

    by_approach = _rows_by_approach(rows)
    required = [baseline]
    for beam, greedy in pairs:
        required.extend([beam, greedy])
    missing = [name for name in required if name not in by_approach]
    if missing:
        return {
            "status": "incomplete",
            "scan_pruning_result": "missing_rows",
            "missing_approaches": missing,
            "recommendation": "rerun identity-history scan-pruning manifest",
        }

    blocks = [
        _pair_block(
            beam_name,
            greedy_name,
            by_approach[beam_name],
            by_approach[greedy_name],
            by_approach[baseline],
            tolerance=float(tolerance),
        )
        for beam_name, greedy_name in pairs
    ]
    viable = [block for block in blocks if block["decision"] == "viable_complete_history_gain"]
    greedy_regressions = [
        block for block in blocks if block["decision"] == "beam_regression_vs_greedy"
    ]
    baseline_regressions = [
        block for block in blocks if block["decision"] == "scan_pruning_regression_vs_identity_history"
    ]
    pairwise_only = [block for block in blocks if block["decision"] == "pairwise_only_gain"]
    ties = [block for block in blocks if block["decision"] == "tie"]
    best = max(
        blocks,
        key=lambda block: (
            float(block["beam_minus_greedy_complete_track_f1_micro"]),
            float(block["beam_minus_baseline_complete_track_f1_micro"]),
            float(block["beam_minus_greedy_pairwise_f1_micro"]),
        ),
    )
    result = _scan_pruning_result(
        n_viable=len(viable),
        n_ties=len(ties),
        n_pairwise_only=len(pairwise_only),
        n_greedy_regressions=len(greedy_regressions),
        n_baseline_regressions=len(baseline_regressions),
        n_pairs=len(blocks),
    )
    return {
        "status": "complete",
        "baseline": baseline,
        "scan_pruning_result": result,
        "best_candidate": str(best["beam"]),
        "viable_candidate_count": int(len(viable)),
        "tie_candidate_count": int(len(ties)),
        "pairwise_only_count": int(len(pairwise_only)),
        "greedy_regression_count": int(len(greedy_regressions)),
        "baseline_regression_count": int(len(baseline_regressions)),
        "recommendation": _recommendation(result),
        "candidate_decisions": blocks,
    }


def format_scan_pruning_decision_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact scan-pruning decision note."""

    if decision.get("status") != "complete":
        missing = ", ".join(str(item) for item in decision.get("missing_approaches", ()))
        return "\n".join(
            [
                "# FullMHT Identity-History Scan-Pruning Decision",
                "",
                "Status: incomplete",
                f"Missing approaches: {missing or 'none reported'}",
                f"Recommendation: {decision.get('recommendation', '')}",
            ]
        )

    lines = [
        "# FullMHT Identity-History Scan-Pruning Decision",
        "",
        f"Result: `{decision['scan_pruning_result']}`",
        f"Best candidate: `{decision['best_candidate']}`",
        f"Recommendation: {decision['recommendation']}",
        "",
        "| candidate | decision | vs greedy pairwise micro | vs greedy complete micro | vs baseline pairwise micro | vs baseline complete micro |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for block in decision.get("candidate_decisions", ()):  # type: ignore[assignment]
        row = dict(block)
        lines.append(
            "| {beam} | {decision} | {g_pair:.6g} | {g_complete:.6g} | {b_pair:.6g} | {b_complete:.6g} |".format(
                beam=row["beam"],
                decision=row["decision"],
                g_pair=float(row["beam_minus_greedy_pairwise_f1_micro"]),
                g_complete=float(row["beam_minus_greedy_complete_track_f1_micro"]),
                b_pair=float(row["beam_minus_baseline_pairwise_f1_micro"]),
                b_complete=float(row["beam_minus_baseline_complete_track_f1_micro"]),
            )
        )
    return "\n".join(lines)


def write_scan_pruning_decision(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write a scan-pruning decision artifact as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(format_scan_pruning_decision_markdown(decision) + "\n", encoding="utf-8")


def _rows_by_approach(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_approach: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        approach = str(row.get("approach", ""))
        if approach:
            by_approach[approach] = row
    return by_approach


def _pair_block(
    beam_name: str,
    greedy_name: str,
    beam_row: Mapping[str, Any],
    greedy_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    block: dict[str, Any] = {"beam": beam_name, "greedy": greedy_name}
    block.update(_delta_block(beam_row, greedy_row, prefix="beam_minus_greedy"))
    block.update(_delta_block(beam_row, baseline_row, prefix="beam_minus_baseline"))
    greedy_deltas = [float(block[f"beam_minus_greedy_{metric}"]) for metric in REPORT_METRICS]
    baseline_deltas = [float(block[f"beam_minus_baseline_{metric}"]) for metric in REPORT_METRICS]
    complete_gain = float(block["beam_minus_greedy_complete_track_f1_micro"])
    pairwise_gain = float(block["beam_minus_greedy_pairwise_f1_micro"])
    if any(delta < -float(tolerance) for delta in greedy_deltas):
        decision = "beam_regression_vs_greedy"
    elif any(delta < -float(tolerance) for delta in baseline_deltas):
        decision = "scan_pruning_regression_vs_identity_history"
    elif complete_gain > float(tolerance):
        decision = "viable_complete_history_gain"
    elif pairwise_gain > float(tolerance):
        decision = "pairwise_only_gain"
    else:
        decision = "tie"
    block["decision"] = decision
    return block


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
        raise ValueError(f"Comparison metric {metric!r} is not numeric: {row.get(metric)!r}") from exc


def _scan_pruning_result(
    *,
    n_viable: int,
    n_ties: int,
    n_pairwise_only: int,
    n_greedy_regressions: int,
    n_baseline_regressions: int,
    n_pairs: int,
) -> str:
    if n_greedy_regressions > 0:
        return "scan_pruning_beam_regression_vs_greedy"
    if n_baseline_regressions > 0:
        return "scan_pruning_regression_vs_identity_history"
    if n_viable >= 2:
        return "scan_pruning_stable_complete_history_gain"
    if n_viable == 1:
        return "scan_pruning_single_weight_gain"
    if n_pairwise_only > 0:
        return "scan_pruning_pairwise_only_gain"
    if n_ties == n_pairs:
        return "scan_pruning_ties_identity_history"
    return "scan_pruning_not_promotable"


def _recommendation(result: str) -> str:
    if result == "scan_pruning_stable_complete_history_gain":
        return "promote scan-history pruning only after full identity-history gates also pass"
    if result == "scan_pruning_single_weight_gain":
        return "treat as exploratory; scan-history gain is knife-edge"
    if result == "scan_pruning_ties_identity_history":
        return "record as method validation but do not promote as better"
    return "do not promote scan-history pruning from this probe"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_identity_history_scan_pruning_decision",
        description="Interpret a FullMHT identity-history scan-pruning comparison CSV.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decision = evaluate_identity_history_scan_pruning_decision(
        load_comparison_rows(args.comparison_csv)
    )
    if args.output is not None:
        write_scan_pruning_decision(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_scan_pruning_decision_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
