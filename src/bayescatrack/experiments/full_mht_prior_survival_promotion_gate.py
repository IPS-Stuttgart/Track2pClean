"""Promotion gate for the FullMHT prior-survival candidate.

The calibrated prior-survival row is the first candidate that can replace the
fixed prior-veto pocket with a label-free likelihood model.  It should not be
promoted from one benchmark table alone.  This helper combines the frozen
canonical manifest decision, the prior-survival sensitivity table, and the
label-free exposure audit into one mechanical decision.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bayescatrack.experiments.full_mht_history_dynamics_promotion_gate import (
    HistoryDynamicsPromotionConfig,
    evaluate_exposure_gate,
    load_exposure_rows,
)
from bayescatrack.experiments.full_mht_manifest_decision import (
    evaluate_full_mht_manifest_decision,
    load_comparison_rows,
)


PRIOR_SURVIVAL_PROMOTABLE_RESULTS = {
    "survival_improves_fixed_veto",
    "survival_ties_fixed_veto",
}


@dataclass(frozen=True)
class PriorSurvivalSensitivityConfig:
    """Frozen row names and stability limits for the sensitivity table."""

    base_row: str = "FullMHTPrior2"
    central_row: str = "SurvivalW10Clip8"
    variant_rows: tuple[str, ...] = (
        "SurvivalW05Clip8",
        "SurvivalW10Clip8",
        "SurvivalW15Clip8",
        "SurvivalW10Clip4",
        "SurvivalW10MinExamples3",
        "SurvivalStrictAnchors",
    )
    weight_variant_rows: tuple[str, ...] = (
        "SurvivalW05Clip8",
        "SurvivalW10Clip8",
        "SurvivalW15Clip8",
    )
    min_passing_variants: int = 4
    min_passing_weight_variants: int = 2
    max_pairwise_drop: float = 0.01
    tolerance: float = 1.0e-12


@dataclass(frozen=True)
class PriorSurvivalPromotionConfig:
    """Predeclared gates for prior-survival promotion."""

    sensitivity: PriorSurvivalSensitivityConfig = field(
        default_factory=PriorSurvivalSensitivityConfig
    )
    max_selected_non_prior_edges_per_subject: int = 3
    max_total_non_prior_edges: int = 10
    max_switched_prior_successors: int = 0
    max_no_prior_successor_continuations: int = 10
    max_gap_reactivated_tracks: int | None = None


def evaluate_prior_survival_promotion(
    canonical_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: PriorSurvivalPromotionConfig | None = None,
) -> dict[str, Any]:
    """Combine manifest, sensitivity, and exposure evidence."""

    cfg = config or PriorSurvivalPromotionConfig()
    manifest = evaluate_full_mht_manifest_decision(canonical_rows)
    sensitivity = evaluate_prior_survival_sensitivity(
        sensitivity_rows,
        config=cfg.sensitivity,
    )
    exposure = evaluate_exposure_gate(
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
            max_gap_reactivated_tracks=cfg.max_gap_reactivated_tracks,
        ),
    )

    manifest_result = str(manifest.get("history_search_result", "incomplete"))
    survival_result = str(manifest.get("prior_survival_result", "incomplete"))
    sensitivity_result = str(sensitivity.get("sensitivity_result", "incomplete"))
    exposure_result = str(exposure.get("exposure_result", "incomplete"))
    manifest_promotable = (
        manifest.get("status") == "complete"
        and manifest_result == "prior_survival_complete_history_advantage"
        and survival_result in PRIOR_SURVIVAL_PROMOTABLE_RESULTS
    )

    if manifest.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun canonical FullMHT manifest"
    elif sensitivity.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun prior-survival sensitivity manifest"
    elif exposure.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun label-free FullMHT exposure audit"
    elif manifest_promotable and sensitivity_result == "stable_plateau" and exposure_result == "bounded_exposure":
        status = "promotable_after_review"
        recommendation = "promote only after recording exact output directories and no-GT test results"
    elif not manifest_promotable:
        status = "not_promotable_manifest"
        recommendation = "keep exploratory; canonical manifest does not prove calibrated history-search advantage"
    elif sensitivity_result != "stable_plateau":
        status = "not_promotable_sensitivity"
        recommendation = "keep exploratory; prior-survival gain is absent or knife-edge"
    elif exposure_result != "bounded_exposure":
        status = "not_promotable_broad_exposure"
        recommendation = "keep exploratory; label-free exposure is too broad"
    else:
        status = "not_promotable"
        recommendation = "keep exploratory; promotion gates failed"

    return {
        "status": status,
        "recommendation": recommendation,
        "manifest_result": manifest_result,
        "prior_survival_result": survival_result,
        "sensitivity_result": sensitivity_result,
        "exposure_result": exposure_result,
        "manifest": manifest,
        "sensitivity": sensitivity,
        "exposure": exposure,
    }


def evaluate_prior_survival_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: PriorSurvivalSensitivityConfig | None = None,
) -> dict[str, Any]:
    """Evaluate whether prior-survival performance is a small plateau."""

    cfg = config or PriorSurvivalSensitivityConfig()
    by_approach = _rows_by_approach(rows)
    required = (cfg.base_row, *cfg.variant_rows)
    missing = [name for name in required if name not in by_approach]
    if missing:
        return {
            "status": "incomplete",
            "sensitivity_result": "missing_rows",
            "missing_approaches": missing,
            "recommendation": "rerun sensitivity manifest with all frozen prior-survival rows",
        }

    base_pairwise = _metric(by_approach[cfg.base_row], "pairwise_f1_micro")
    base_complete = _metric(by_approach[cfg.base_row], "complete_track_f1_micro")
    passing: list[str] = []
    weight_passing: list[str] = []
    pairwise_collapse: list[str] = []
    deltas: dict[str, dict[str, float]] = {}
    for name in cfg.variant_rows:
        row = by_approach[name]
        pairwise_delta = _metric(row, "pairwise_f1_micro") - base_pairwise
        complete_delta = _metric(row, "complete_track_f1_micro") - base_complete
        deltas[name] = {
            "pairwise_f1_micro_delta_vs_base": float(pairwise_delta),
            "complete_track_f1_micro_delta_vs_base": float(complete_delta),
        }
        if pairwise_delta < -float(cfg.max_pairwise_drop):
            pairwise_collapse.append(name)
        if pairwise_delta >= -float(cfg.tolerance) and complete_delta >= -float(cfg.tolerance):
            passing.append(name)
            if name in cfg.weight_variant_rows:
                weight_passing.append(name)

    central_passes = cfg.central_row in passing
    if pairwise_collapse:
        result = "pairwise_collapse"
    elif not central_passes:
        result = "central_candidate_not_stable"
    elif len(passing) < int(cfg.min_passing_variants):
        result = "sensitivity_not_stable"
    elif len(weight_passing) < int(cfg.min_passing_weight_variants):
        result = "weight_neighborhood_not_stable"
    else:
        result = "stable_plateau"

    return {
        "status": "complete",
        "sensitivity_result": result,
        "base_row": cfg.base_row,
        "central_row": cfg.central_row,
        "passing_variants": passing,
        "passing_weight_variants": weight_passing,
        "pairwise_collapse_variants": pairwise_collapse,
        "n_passing_variants": int(len(passing)),
        "n_required_passing_variants": int(cfg.min_passing_variants),
        "n_passing_weight_variants": int(len(weight_passing)),
        "n_required_passing_weight_variants": int(cfg.min_passing_weight_variants),
        "deltas": deltas,
    }


def format_prior_survival_promotion_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact prior-survival promotion-gate note."""

    sensitivity = dict(decision.get("sensitivity", {}))
    exposure = dict(decision.get("exposure", {}))
    lines = [
        "# FullMHT Prior-Survival Promotion Gate",
        "",
        f"Status: `{decision.get('status', '')}`",
        f"Manifest result: `{decision.get('manifest_result', '')}`",
        f"Prior-survival result: `{decision.get('prior_survival_result', '')}`",
        f"Sensitivity result: `{decision.get('sensitivity_result', '')}`",
        f"Exposure result: `{decision.get('exposure_result', '')}`",
        f"Recommendation: {decision.get('recommendation', '')}",
        "",
        "| sensitivity metric | value | required |",
        "| --- | ---: | ---: |",
        "| passing variants | {value} | {limit} |".format(
            value=sensitivity.get("n_passing_variants", ""),
            limit=sensitivity.get("n_required_passing_variants", ""),
        ),
        "| passing weight variants | {value} | {limit} |".format(
            value=sensitivity.get("n_passing_weight_variants", ""),
            limit=sensitivity.get("n_required_passing_weight_variants", ""),
        ),
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
        ("history_gap_reactivated_tracks", "limit_history_gap_reactivated_tracks"),
    ):
        lines.append(
            "| {metric} | {value} | {limit} |".format(
                metric=metric,
                value=exposure.get(metric, ""),
                limit=exposure.get(limit_key, ""),
            )
        )
    failed = ", ".join(str(item) for item in exposure.get("failed_limits", ()))
    lines.extend(
        [
            "",
            "Passing sensitivity variants: {variants}".format(
                variants=", ".join(str(item) for item in sensitivity.get("passing_variants", ()))
                or "none"
            ),
            f"Pairwise-collapse variants: {', '.join(str(item) for item in sensitivity.get('pairwise_collapse_variants', ())) or 'none'}",
            f"Failed exposure limits: {failed or 'none'}",
        ]
    )
    return "\n".join(lines)


def write_prior_survival_promotion(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write the promotion gate as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(
        format_prior_survival_promotion_markdown(decision) + "\n",
        encoding="utf-8",
    )


def _rows_by_approach(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_approach: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        approach = str(row.get("approach", ""))
        if approach:
            by_approach[approach] = row
    return by_approach


def _metric(row: Mapping[str, Any], metric: str) -> float:
    try:
        return float(row[metric])
    except KeyError as exc:
        raise ValueError(f"Comparison row is missing metric column {metric!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Comparison metric {metric!r} is not numeric: {row.get(metric)!r}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_prior_survival_promotion_gate",
        description="Combine FullMHT prior-survival manifest, sensitivity, and exposure gates.",
    )
    parser.add_argument("canonical_comparison_csv", type=Path)
    parser.add_argument("sensitivity_comparison_csv", type=Path)
    parser.add_argument("exposure_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--min-passing-variants", type=int, default=4)
    parser.add_argument("--min-passing-weight-variants", type=int, default=2)
    parser.add_argument("--max-pairwise-drop", type=float, default=0.01)
    parser.add_argument("--max-non-prior-per-subject", type=int, default=3)
    parser.add_argument("--max-total-non-prior", type=int, default=10)
    parser.add_argument("--max-switches", type=int, default=0)
    parser.add_argument("--max-no-prior-continuations", type=int, default=10)
    parser.add_argument("--max-gap-reactivations", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = PriorSurvivalPromotionConfig(
        sensitivity=PriorSurvivalSensitivityConfig(
            min_passing_variants=max(0, int(args.min_passing_variants)),
            min_passing_weight_variants=max(0, int(args.min_passing_weight_variants)),
            max_pairwise_drop=max(0.0, float(args.max_pairwise_drop)),
        ),
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
        max_gap_reactivated_tracks=(
            None
            if args.max_gap_reactivations is None
            else max(0, int(args.max_gap_reactivations))
        ),
    )
    decision = evaluate_prior_survival_promotion(
        load_comparison_rows(args.canonical_comparison_csv),
        load_comparison_rows(args.sensitivity_comparison_csv),
        load_exposure_rows(args.exposure_csv),
        config=cfg,
    )
    if args.output is not None:
        write_prior_survival_promotion(
            decision,
            args.output,
            output_format=str(args.format),
        )
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_prior_survival_promotion_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
