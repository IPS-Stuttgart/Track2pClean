"""Subject-level support gate for the FullMHT identity-history candidate.

The aggregate identity-history gate can show that the full beam beats its greedy
ablation overall.  This helper asks a different paper-facing question: is that
complete-history gain supported by multiple subjects without hiding a per-subject
regression against the matching greedy or required control rows?
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MetricName = Literal["pairwise_f1", "complete_track_f1"]
SUBJECT_METRICS: tuple[MetricName, ...] = ("pairwise_f1", "complete_track_f1")


@dataclass(frozen=True)
class SubjectSupportConfig:
    """Frozen row names and support thresholds for subject-level validation."""

    candidate: str = "FullMHTIdentityHistory"
    greedy: str = "FullMHTGreedyIdentityHistory"
    controls: tuple[str, ...] = (
        "Track2p",
        "FullMHTPrior2",
        "FullMHTPriorSurvival",
        "FullMHTNoPriorContinuation100",
        "FullMHTIdentityHistoryNoLocalContext",
    )
    min_complete_gain_subjects: int = 2
    max_regressing_subjects: int = 0
    tolerance: float = 1.0e-12


def parse_labeled_input(value: str) -> tuple[str, Path]:
    """Parse a ``LABEL=CSV`` command-line input."""

    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"Input must have form LABEL=CSV, got {value!r}")
    return label, Path(path)


def load_labeled_subject_rows(inputs: Sequence[tuple[str, Path]]) -> list[dict[str, str]]:
    """Load subject benchmark rows and attach approach labels."""

    rows: list[dict[str, str]] = []
    for label, path in inputs:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "subject" not in reader.fieldnames:
                raise ValueError(f"Subject CSV {path} is missing a 'subject' column")
            missing_metrics = [metric for metric in SUBJECT_METRICS if metric not in reader.fieldnames]
            if missing_metrics:
                raise ValueError(
                    f"Subject CSV {path} is missing metric columns: {', '.join(missing_metrics)}"
                )
            for row in reader:
                rows.append({"approach": label, **row})
    if not rows:
        raise ValueError("No subject rows were loaded")
    return rows


def evaluate_subject_support(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: SubjectSupportConfig | None = None,
) -> dict[str, Any]:
    """Evaluate subject-level support for the identity-history candidate."""

    cfg = config or SubjectSupportConfig()
    required = (cfg.candidate, cfg.greedy, *cfg.controls)
    by_subject = _rows_by_subject_and_approach(rows)
    missing = _missing_subject_approaches(by_subject, required)
    if missing:
        return {
            "status": "incomplete",
            "subject_support_result": "missing_subject_rows",
            "missing_subject_approaches": missing,
            "recommendation": "rerun subject-support decision with all frozen identity-history row CSVs",
        }

    subject_blocks = [
        _subject_block(subject, subject_rows, config=cfg)
        for subject, subject_rows in sorted(by_subject.items())
    ]
    complete_gain_subjects = [
        block["subject"] for block in subject_blocks if block["complete_gain_vs_greedy"] == "true"
    ]
    greedy_regression_subjects = [
        block["subject"] for block in subject_blocks if block["regresses_vs_greedy"] == "true"
    ]
    control_regression_subjects = [
        block["subject"] for block in subject_blocks if block["regresses_vs_control"] == "true"
    ]
    regression_subjects = sorted(set(greedy_regression_subjects) | set(control_regression_subjects))
    worst_complete_delta = min(
        float(block["candidate_minus_greedy_complete_track_f1"]) for block in subject_blocks
    )
    worst_pairwise_delta = min(
        float(block["candidate_minus_greedy_pairwise_f1"]) for block in subject_blocks
    )
    worst_control_delta = min(float(block["worst_control_delta"]) for block in subject_blocks)

    if len(regression_subjects) > int(cfg.max_regressing_subjects):
        result = "subject_metric_regression"
    elif len(complete_gain_subjects) >= int(cfg.min_complete_gain_subjects):
        result = "stable_subject_support"
    elif complete_gain_subjects:
        result = "weak_subject_support"
    else:
        result = "no_subject_complete_gain"

    return {
        "status": "complete",
        "subject_support_result": result,
        "candidate": cfg.candidate,
        "greedy": cfg.greedy,
        "controls": list(cfg.controls),
        "subjects": int(len(subject_blocks)),
        "complete_gain_subjects": complete_gain_subjects,
        "greedy_regression_subjects": greedy_regression_subjects,
        "control_regression_subjects": control_regression_subjects,
        "regression_subjects": regression_subjects,
        "n_complete_gain_subjects": int(len(complete_gain_subjects)),
        "n_regressing_subjects": int(len(regression_subjects)),
        "min_complete_gain_subjects": int(cfg.min_complete_gain_subjects),
        "max_regressing_subjects": int(cfg.max_regressing_subjects),
        "worst_candidate_minus_greedy_complete_track_f1": float(worst_complete_delta),
        "worst_candidate_minus_greedy_pairwise_f1": float(worst_pairwise_delta),
        "worst_candidate_minus_control_metric": float(worst_control_delta),
        "subject_deltas": subject_blocks,
        "recommendation": _recommendation(result),
    }


def format_subject_support_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact subject-support decision note."""

    if decision.get("status") != "complete":
        missing = decision.get("missing_subject_approaches", {})
        return "\n".join(
            [
                "# FullMHT Identity-History Subject Support Decision",
                "",
                "Status: incomplete",
                f"Result: `{decision.get('subject_support_result', '')}`",
                f"Missing subject approaches: {json.dumps(missing, sort_keys=True)}",
                f"Recommendation: {decision.get('recommendation', '')}",
            ]
        )

    lines = [
        "# FullMHT Identity-History Subject Support Decision",
        "",
        f"Status: `{decision.get('status', '')}`",
        f"Result: `{decision.get('subject_support_result', '')}`",
        f"Candidate: `{decision.get('candidate', '')}`",
        f"Greedy baseline: `{decision.get('greedy', '')}`",
        "Complete-gain subjects: {subjects}".format(
            subjects=", ".join(str(item) for item in decision.get("complete_gain_subjects", ()))
            or "none"
        ),
        "Regression subjects: {subjects}".format(
            subjects=", ".join(str(item) for item in decision.get("regression_subjects", ()))
            or "none"
        ),
        f"Recommendation: {decision.get('recommendation', '')}",
        "",
        "| subject | complete delta vs greedy | pairwise delta vs greedy | worst control delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for block in decision.get("subject_deltas", ()):  # type: ignore[assignment]
        subject_block = dict(block)
        lines.append(
            "| {subject} | {complete:.6g} | {pairwise:.6g} | {control:.6g} |".format(
                subject=subject_block.get("subject", ""),
                complete=float(subject_block.get("candidate_minus_greedy_complete_track_f1", 0.0)),
                pairwise=float(subject_block.get("candidate_minus_greedy_pairwise_f1", 0.0)),
                control=float(subject_block.get("worst_control_delta", 0.0)),
            )
        )
    return "\n".join(lines)


def write_subject_support(decision: Mapping[str, Any], output: Path, *, output_format: str) -> None:
    """Write the subject-support decision as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(format_subject_support_markdown(decision) + "\n", encoding="utf-8")


def _rows_by_subject_and_approach(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    by_subject: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        subject = str(row.get("subject", ""))
        approach = str(row.get("approach", ""))
        if not subject or not approach:
            continue
        by_subject.setdefault(subject, {})[approach] = row
    return by_subject


def _missing_subject_approaches(
    by_subject: Mapping[str, Mapping[str, Mapping[str, Any]]],
    required: Sequence[str],
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    if not by_subject:
        return {"<none>": list(required)}
    for subject, subject_rows in sorted(by_subject.items()):
        absent = [approach for approach in required if approach not in subject_rows]
        if absent:
            missing[subject] = absent
    return missing


def _subject_block(
    subject: str,
    subject_rows: Mapping[str, Mapping[str, Any]],
    *,
    config: SubjectSupportConfig,
) -> dict[str, Any]:
    candidate = subject_rows[config.candidate]
    greedy = subject_rows[config.greedy]
    candidate_minus_greedy = {
        metric: _metric(candidate, metric) - _metric(greedy, metric)
        for metric in SUBJECT_METRICS
    }
    control_deltas: dict[str, dict[str, float]] = {}
    for control in config.controls:
        control_row = subject_rows[control]
        control_deltas[control] = {
            metric: _metric(candidate, metric) - _metric(control_row, metric)
            for metric in SUBJECT_METRICS
        }
    worst_control_delta = min(
        delta for deltas in control_deltas.values() for delta in deltas.values()
    )
    regresses_vs_greedy = any(
        delta < -float(config.tolerance) for delta in candidate_minus_greedy.values()
    )
    regresses_vs_control = worst_control_delta < -float(config.tolerance)
    complete_gain = (
        candidate_minus_greedy["complete_track_f1"] > float(config.tolerance)
        and candidate_minus_greedy["pairwise_f1"] >= -float(config.tolerance)
    )
    return {
        "subject": subject,
        "candidate_minus_greedy_pairwise_f1": float(candidate_minus_greedy["pairwise_f1"]),
        "candidate_minus_greedy_complete_track_f1": float(
            candidate_minus_greedy["complete_track_f1"]
        ),
        "worst_control_delta": float(worst_control_delta),
        "complete_gain_vs_greedy": _format_bool(complete_gain),
        "regresses_vs_greedy": _format_bool(regresses_vs_greedy),
        "regresses_vs_control": _format_bool(regresses_vs_control),
        "control_deltas": control_deltas,
    }


def _metric(row: Mapping[str, Any], metric: MetricName) -> float:
    try:
        return float(row[metric])
    except KeyError as exc:
        raise ValueError(f"Subject row is missing metric column {metric!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Subject metric {metric!r} is not numeric: {row.get(metric)!r}") from exc


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _recommendation(result: str) -> str:
    if result == "stable_subject_support":
        return "treat subject-level support as compatible with promotion after aggregate gates pass"
    if result == "weak_subject_support":
        return "keep exploratory; aggregate gain is supported by too few subjects"
    if result == "subject_metric_regression":
        return "keep exploratory; candidate hides a subject-level regression"
    if result == "no_subject_complete_gain":
        return "keep exploratory; no subject shows complete-track gain over greedy"
    return "rerun subject-support decision before interpreting"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_identity_history_subject_support_decision",
        description="Evaluate subject-level support for FullMHT identity-history promotion.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Labeled subject benchmark CSV to include; repeat for each frozen approach",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--min-complete-gain-subjects", type=int, default=2)
    parser.add_argument("--max-regressing-subjects", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    inputs = [parse_labeled_input(value) for value in args.input]
    decision = evaluate_subject_support(
        load_labeled_subject_rows(inputs),
        config=SubjectSupportConfig(
            min_complete_gain_subjects=max(0, int(args.min_complete_gain_subjects)),
            max_regressing_subjects=max(0, int(args.max_regressing_subjects)),
            tolerance=max(0.0, float(args.tolerance)),
        ),
    )
    if args.output is not None:
        write_subject_support(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_subject_support_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
