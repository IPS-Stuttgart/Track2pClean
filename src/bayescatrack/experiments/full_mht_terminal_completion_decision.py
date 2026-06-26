"""Decision helper for FullMHT terminal-completion probe manifests.

The terminal-completion objective is a label-free complete-history penalty used by
FullMHT terminal reranking.  This module does not tune that objective.  It reads
the frozen comparison CSV from a small manifest and applies a predeclared rule:
complete-track F1 must improve without pairwise-F1 regression, and the gain must
not appear at only one isolated weight.
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
class TerminalCompletionDecisionConfig:
    """Frozen row names and tolerances for terminal-completion probes."""

    baseline: str = "FullMHTPrior2"
    candidates: tuple[str, ...] = (
        "FullMHTTerminalCompletion025",
        "FullMHTTerminalCompletion050",
        "FullMHTTerminalCompletion100",
    )
    track2p: str = "Track2p"
    tolerance: float = 1e-12


def load_comparison_rows(path: Path) -> list[dict[str, str]]:
    """Load benchmark comparison rows keyed by ``approach``."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No comparison rows found in {path}")
    if "approach" not in rows[0]:
        raise ValueError(f"Comparison CSV {path} is missing an 'approach' column")
    return rows


def evaluate_terminal_completion_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: TerminalCompletionDecisionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate whether terminal-completion weighting is stable enough to promote."""

    cfg = config or TerminalCompletionDecisionConfig()
    by_approach = _rows_by_approach(rows)
    required = (cfg.baseline, *cfg.candidates)
    missing = [name for name in required if name not in by_approach]
    if missing:
        return {
            "status": "incomplete",
            "missing_approaches": missing,
            "recommendation": "rerun terminal-completion probe manifest",
        }

    baseline = by_approach[cfg.baseline]
    candidate_blocks = [
        _candidate_block(
            name,
            by_approach[name],
            baseline,
            tolerance=float(cfg.tolerance),
        )
        for name in cfg.candidates
    ]
    viable = [block for block in candidate_blocks if block["decision"] == "viable_gain"]
    ties = [block for block in candidate_blocks if block["decision"] == "tie"]
    pairwise_regressions = [
        block for block in candidate_blocks if block["decision"] == "pairwise_regression"
    ]
    complete_regressions = [
        block for block in candidate_blocks if block["decision"] == "complete_regression"
    ]
    best = max(
        candidate_blocks,
        key=lambda block: (
            float(block["delta_complete_track_f1_micro"]),
            float(block["delta_pairwise_f1_micro"]),
        ),
    )
    result = _terminal_result(
        n_viable=len(viable),
        n_ties=len(ties),
        n_pairwise_regressions=len(pairwise_regressions),
        n_complete_regressions=len(complete_regressions),
        n_candidates=len(candidate_blocks),
    )
    decision: dict[str, Any] = {
        "status": "complete",
        "baseline": cfg.baseline,
        "candidates": list(cfg.candidates),
        "terminal_completion_result": result,
        "best_candidate": str(best["approach"]),
        "best_candidate_complete_track_f1_micro_delta": float(
            best["delta_complete_track_f1_micro"]
        ),
        "best_candidate_pairwise_f1_micro_delta": float(
            best["delta_pairwise_f1_micro"]
        ),
        "viable_candidate_count": int(len(viable)),
        "tie_candidate_count": int(len(ties)),
        "pairwise_regression_count": int(len(pairwise_regressions)),
        "complete_regression_count": int(len(complete_regressions)),
        "recommendation": _recommendation(result),
        "candidate_decisions": candidate_blocks,
    }
    if cfg.track2p in by_approach:
        decision.update(
            _delta_block(
                by_approach[cfg.baseline],
                by_approach[cfg.track2p],
                prefix="baseline_minus_track2p",
            )
        )
        decision.update(
            _delta_block(
                by_approach[str(best["approach"])],
                by_approach[cfg.track2p],
                prefix="best_minus_track2p",
            )
        )
    return decision


def format_decision_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact terminal-completion decision note."""

    if decision.get("status") != "complete":
        missing = ", ".join(str(item) for item in decision.get("missing_approaches", ()))
        return "\n".join(
            [
                "# FullMHT Terminal Completion Decision",
                "",
                "Status: incomplete",
                f"Missing approaches: {missing or 'none reported'}",
                f"Recommendation: {decision.get('recommendation', '')}",
            ]
        )

    lines = [
        "# FullMHT Terminal Completion Decision",
        "",
        f"Result: `{decision['terminal_completion_result']}`",
        f"Best candidate: `{decision['best_candidate']}`",
        f"Recommendation: {decision['recommendation']}",
        "",
        "| candidate | decision | pairwise F1 micro delta | complete-track F1 micro delta |",
        "| --- | --- | ---: | ---: |",
    ]
    for block in decision.get("candidate_decisions", ()):  # type: ignore[assignment]
        row = dict(block)
        lines.append(
            "| {approach} | {decision} | {pairwise:.6g} | {complete:.6g} |".format(
                approach=row["approach"],
                decision=row["decision"],
                pairwise=float(row["delta_pairwise_f1_micro"]),
                complete=float(row["delta_complete_track_f1_micro"]),
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


def _candidate_block(
    approach: str,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    block: dict[str, Any] = {"approach": approach}
    block.update(_delta_block(candidate, baseline, prefix="delta"))
    pairwise_delta = float(block["delta_pairwise_f1_micro"])
    complete_delta = float(block["delta_complete_track_f1_micro"])
    if pairwise_delta < -float(tolerance):
        decision = "pairwise_regression"
    elif complete_delta < -float(tolerance):
        decision = "complete_regression"
    elif complete_delta > float(tolerance):
        decision = "viable_gain"
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
        raise ValueError(f"Comparison metric {metric!r} is not numeric: {row[metric]!r}") from exc


def _terminal_result(
    *,
    n_viable: int,
    n_ties: int,
    n_pairwise_regressions: int,
    n_complete_regressions: int,
    n_candidates: int,
) -> str:
    if n_viable >= 2:
        return "terminal_completion_stable_gain"
    if n_viable == 1:
        return "terminal_completion_single_weight_gain"
    if n_ties == n_candidates:
        return "terminal_completion_ties_baseline"
    if n_pairwise_regressions > 0:
        return "terminal_completion_pairwise_regression"
    if n_complete_regressions > 0:
        return "terminal_completion_complete_regression"
    return "terminal_completion_not_promotable"


def _recommendation(result: str) -> str:
    if result == "terminal_completion_stable_gain":
        return "promote only after exposure/no-GT gates also pass"
    if result == "terminal_completion_single_weight_gain":
        return "treat as exploratory; gain is knife-edge in the frozen probe"
    if result == "terminal_completion_ties_baseline":
        return "record as complete-history validation but do not promote as better"
    return "do not promote terminal completion from this probe"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_terminal_completion_decision",
        description="Summarize terminal-completion FullMHT probe comparison rows.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decision = evaluate_terminal_completion_decision(
        load_comparison_rows(args.comparison_csv)
    )
    if args.output is not None:
        write_decision(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_decision_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
