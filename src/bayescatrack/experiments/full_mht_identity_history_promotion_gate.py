"""Promotion gate for the combined FullMHT identity-history candidate.

The identity-history row combines calibrated association likelihood, calibrated
Track2p prior-edge survival, no-prior continuation likelihood, and
growth-history prediction.  It should be promoted only if three independent
artifacts agree: the frozen manifest shows beam-over-greedy complete-history
advantage, the immediate sensitivity neighborhood is stable, and the label-free
exposure audit shows the enabled model layers remain rare and active.
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
from bayescatrack.experiments.full_mht_identity_history_decision import (
    evaluate_identity_history_decision,
    load_comparison_rows,
)

IDENTITY_HISTORY_EXPOSURE_COLUMNS = (
    "history_prior_survival_scored_edges",
    "history_prior_survival_positive_edges",
    "history_prior_survival_negative_edges",
    "max_prior_survival_negative_edges_per_subject",
    "history_no_prior_continuation_scored_edges",
    "history_no_prior_continuation_positive_edges",
    "history_no_prior_continuation_negative_edges",
    "max_no_prior_continuation_positive_edges_per_subject",
    "max_no_prior_continuation_abs_weighted_score_per_subject",
    "history_growth_prediction_evaluated_edges",
    "history_growth_prediction_penalized_edges",
    "max_growth_prediction_penalized_edges_per_subject",
    "max_growth_prediction_weighted_penalty_per_subject",
)


@dataclass(frozen=True)
class IdentityHistorySensitivityConfig:
    """Frozen row names and stability limits for the combined candidate."""

    base_row: str = "FullMHTPrior2"
    central_row: str = "IdentityHistoryCentral"
    variant_rows: tuple[str, ...] = (
        "IdentityHistorySurvivalW05",
        "IdentityHistoryCentral",
        "IdentityHistorySurvivalW15",
        "IdentityHistoryNoPriorW05",
        "IdentityHistoryNoPriorW15",
        "IdentityHistoryGrowthW025",
        "IdentityHistoryGrowthW100",
    )
    survival_variant_rows: tuple[str, ...] = (
        "IdentityHistorySurvivalW05",
        "IdentityHistoryCentral",
        "IdentityHistorySurvivalW15",
    )
    no_prior_variant_rows: tuple[str, ...] = (
        "IdentityHistoryNoPriorW05",
        "IdentityHistoryCentral",
        "IdentityHistoryNoPriorW15",
    )
    growth_variant_rows: tuple[str, ...] = (
        "IdentityHistoryGrowthW025",
        "IdentityHistoryCentral",
        "IdentityHistoryGrowthW100",
    )
    min_passing_variants: int = 5
    min_passing_axis_variants: int = 2
    max_pairwise_drop: float = 0.01
    tolerance: float = 1.0e-12


@dataclass(frozen=True)
class IdentityHistoryPromotionConfig:
    """Predeclared gates for identity-history promotion."""

    sensitivity: IdentityHistorySensitivityConfig = field(default_factory=IdentityHistorySensitivityConfig)
    max_selected_non_prior_edges_per_subject: int = 3
    max_total_non_prior_edges: int = 10
    max_switched_prior_successors: int = 0
    max_no_prior_successor_continuations: int = 10
    max_gap_reactivated_tracks: int | None = None
    max_prior_survival_negative_edges_per_subject: int = 3
    max_total_prior_survival_negative_edges: int = 10
    max_no_prior_continuation_positive_edges_per_subject: int = 3
    max_total_no_prior_continuation_negative_edges: int = 10
    max_no_prior_continuation_abs_weighted_score_per_subject: float = 8.0
    max_growth_prediction_penalized_edges_per_subject: int = 3
    max_total_growth_prediction_penalized_edges: int = 10
    max_growth_prediction_weighted_penalty_per_subject: float = 8.0
    require_active_layer_signals: bool = True


def evaluate_identity_history_promotion(
    canonical_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: IdentityHistoryPromotionConfig | None = None,
) -> dict[str, Any]:
    """Combine frozen comparison, sensitivity, and exposure evidence."""

    cfg = config or IdentityHistoryPromotionConfig()
    manifest = evaluate_identity_history_decision(canonical_rows)
    sensitivity = evaluate_identity_history_sensitivity(sensitivity_rows, config=cfg.sensitivity)
    exposure = evaluate_identity_history_exposure(exposure_rows, config=cfg)

    history_result = str(manifest.get("history_search_result", "incomplete"))
    mht_vs_local_result = str(manifest.get("mht_vs_local_result", history_result))
    prior_result = str(manifest.get("prior_control_result", "incomplete"))
    track2p_result = str(manifest.get("track2p_control_result", "incomplete"))
    no_local_result = str(manifest.get("no_local_context_control_result", "incomplete"))
    layer_result = str(manifest.get("layer_combination_result", "incomplete"))
    sensitivity_result = str(sensitivity.get("sensitivity_result", "incomplete"))
    exposure_result = str(exposure.get("exposure_result", "incomplete"))
    manifest_promotable = (
        manifest.get("status") == "complete"
        and mht_vs_local_result == "identity_complete_history_advantage"
        and prior_result != "identity_below_prior"
        and track2p_result != "identity_below_track2p"
        and no_local_result != "identity_below_no_local_context"
        and layer_result != "combined_layer_regression"
    )

    if manifest.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun identity-history candidate manifest"
    elif sensitivity.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun identity-history sensitivity manifest"
    elif exposure.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun label-free FullMHT exposure audit with identity-history layers enabled"
    elif manifest_promotable and sensitivity_result == "stable_plateau" and exposure_result == "bounded_exposure":
        status = "promotable_after_review"
        recommendation = "promote only after recording exact output directories and no-GT test results"
    elif not manifest_promotable:
        status = "not_promotable_manifest"
        recommendation = "keep exploratory; manifest does not prove MHT-vs-local identity-history advantage"
    elif sensitivity_result != "stable_plateau":
        status = "not_promotable_sensitivity"
        recommendation = "keep exploratory; identity-history gain is absent or knife-edge"
    elif exposure_result != "bounded_exposure":
        status = "not_promotable_broad_exposure"
        recommendation = "keep exploratory; label-free identity-history exposure is too broad"
    else:
        status = "not_promotable"
        recommendation = "keep exploratory; promotion gates failed"

    return {
        "status": status,
        "recommendation": recommendation,
        "mht_vs_local_result": mht_vs_local_result,
        "history_search_result": history_result,
        "prior_control_result": prior_result,
        "track2p_control_result": track2p_result,
        "no_local_context_control_result": no_local_result,
        "layer_combination_result": layer_result,
        "sensitivity_result": sensitivity_result,
        "exposure_result": exposure_result,
        "manifest": manifest,
        "sensitivity": sensitivity,
        "exposure": exposure,
    }


def evaluate_identity_history_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: IdentityHistorySensitivityConfig | None = None,
) -> dict[str, Any]:
    """Evaluate whether the combined candidate has a small stable neighborhood."""

    cfg = config or IdentityHistorySensitivityConfig()
    by_approach = _rows_by_approach(rows)
    required = (cfg.base_row, *_unique(cfg.variant_rows))
    missing = [name for name in required if name not in by_approach]
    if missing:
        return {
            "status": "incomplete",
            "sensitivity_result": "missing_rows",
            "missing_approaches": missing,
            "recommendation": "rerun sensitivity manifest with all frozen identity-history rows",
        }

    base_pairwise = _metric(by_approach[cfg.base_row], "pairwise_f1_micro")
    base_complete = _metric(by_approach[cfg.base_row], "complete_track_f1_micro")
    passing: list[str] = []
    pairwise_collapse: list[str] = []
    deltas: dict[str, dict[str, float]] = {}
    for name in _unique(cfg.variant_rows):
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

    axis_counts = {
        "survival": _passing_axis_count(cfg.survival_variant_rows, passing),
        "no_prior": _passing_axis_count(cfg.no_prior_variant_rows, passing),
        "growth": _passing_axis_count(cfg.growth_variant_rows, passing),
    }
    central_passes = cfg.central_row in passing
    weak_axes = [name for name, count in axis_counts.items() if count < int(cfg.min_passing_axis_variants)]
    if pairwise_collapse:
        result = "pairwise_collapse"
    elif not central_passes:
        result = "central_candidate_not_stable"
    elif len(passing) < int(cfg.min_passing_variants):
        result = "sensitivity_not_stable"
    elif weak_axes:
        result = "axis_neighborhood_not_stable"
    else:
        result = "stable_plateau"

    return {
        "status": "complete",
        "sensitivity_result": result,
        "base_row": cfg.base_row,
        "central_row": cfg.central_row,
        "passing_variants": passing,
        "pairwise_collapse_variants": pairwise_collapse,
        "axis_passing_counts": axis_counts,
        "weak_axes": weak_axes,
        "n_passing_variants": int(len(passing)),
        "n_required_passing_variants": int(cfg.min_passing_variants),
        "n_required_passing_axis_variants": int(cfg.min_passing_axis_variants),
        "deltas": deltas,
    }


def evaluate_identity_history_exposure(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: IdentityHistoryPromotionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate generic FullMHT exposure and the three enabled model signals."""

    cfg = config or IdentityHistoryPromotionConfig()
    base = evaluate_exposure_gate(
        rows,
        config=HistoryDynamicsPromotionConfig(
            max_selected_non_prior_edges_per_subject=int(cfg.max_selected_non_prior_edges_per_subject),
            max_total_non_prior_edges=int(cfg.max_total_non_prior_edges),
            max_switched_prior_successors=int(cfg.max_switched_prior_successors),
            max_no_prior_successor_continuations=int(cfg.max_no_prior_successor_continuations),
            max_gap_reactivated_tracks=cfg.max_gap_reactivated_tracks,
        ),
    )
    if base.get("status") != "complete":
        return base
    all_row = _all_row(rows)
    if all_row is None:
        return {
            "status": "incomplete",
            "exposure_result": "missing_all_row",
            "recommendation": "rerun exposure audit with aggregate ALL row",
        }
    missing_columns = [key for key in IDENTITY_HISTORY_EXPOSURE_COLUMNS if key not in all_row]
    if missing_columns:
        return {
            "status": "incomplete",
            "exposure_result": "missing_identity_history_exposure_columns",
            "missing_columns": missing_columns,
            "recommendation": "rerun exposure audit with identity-history scoring enabled",
        }

    failures = list(base.get("failed_limits", ()))
    prior_scored = _int_metric(all_row, "history_prior_survival_scored_edges")
    prior_positive = _int_metric(all_row, "history_prior_survival_positive_edges")
    prior_negative = _int_metric(all_row, "history_prior_survival_negative_edges")
    prior_active = int(prior_positive) + int(prior_negative)
    max_prior_negative = _int_metric(all_row, "max_prior_survival_negative_edges_per_subject")
    no_prior_scored = _int_metric(all_row, "history_no_prior_continuation_scored_edges")
    no_prior_positive = _int_metric(all_row, "history_no_prior_continuation_positive_edges")
    no_prior_negative = _int_metric(all_row, "history_no_prior_continuation_negative_edges")
    no_prior_active = int(no_prior_positive) + int(no_prior_negative)
    max_no_prior_positive = _int_metric(all_row, "max_no_prior_continuation_positive_edges_per_subject")
    max_no_prior_abs = _float_metric(all_row, "max_no_prior_continuation_abs_weighted_score_per_subject")
    growth_evaluated = _int_metric(all_row, "history_growth_prediction_evaluated_edges")
    growth_penalized = _int_metric(all_row, "history_growth_prediction_penalized_edges")
    max_growth_penalized = _int_metric(all_row, "max_growth_prediction_penalized_edges_per_subject")
    max_growth_weighted = _float_metric(all_row, "max_growth_prediction_weighted_penalty_per_subject")

    if bool(cfg.require_active_layer_signals):
        inactive: list[str] = []
        if prior_scored <= 0 or prior_active <= 0:
            inactive.append("prior_survival")
        if no_prior_scored <= 0 or no_prior_active <= 0:
            inactive.append("no_prior_continuation")
        if growth_evaluated <= 0 or growth_penalized <= 0:
            inactive.append("growth_history_prediction")
        if inactive:
            return {
                "status": "incomplete",
                "exposure_result": "identity_history_layers_not_active",
                "inactive_layers": inactive,
                "recommendation": "rerun exposure audit; enabled identity-history layers did not produce active signals",
            }

    if max_prior_negative > int(cfg.max_prior_survival_negative_edges_per_subject):
        failures.append("max_prior_survival_negative_edges_per_subject")
    if prior_negative > int(cfg.max_total_prior_survival_negative_edges):
        failures.append("history_prior_survival_negative_edges")
    if max_no_prior_positive > int(cfg.max_no_prior_continuation_positive_edges_per_subject):
        failures.append("max_no_prior_continuation_positive_edges_per_subject")
    if no_prior_negative > int(cfg.max_total_no_prior_continuation_negative_edges):
        failures.append("history_no_prior_continuation_negative_edges")
    if max_no_prior_abs > float(cfg.max_no_prior_continuation_abs_weighted_score_per_subject):
        failures.append("max_no_prior_continuation_abs_weighted_score_per_subject")
    if max_growth_penalized > int(cfg.max_growth_prediction_penalized_edges_per_subject):
        failures.append("max_growth_prediction_penalized_edges_per_subject")
    if growth_penalized > int(cfg.max_total_growth_prediction_penalized_edges):
        failures.append("history_growth_prediction_penalized_edges")
    if max_growth_weighted > float(cfg.max_growth_prediction_weighted_penalty_per_subject):
        failures.append("max_growth_prediction_weighted_penalty_per_subject")

    output = dict(base)
    output.update(
        {
            "exposure_result": "bounded_exposure" if not failures else "broad_exposure",
            "failed_limits": failures,
            "history_prior_survival_scored_edges": int(prior_scored),
            "history_prior_survival_active_edges": int(prior_active),
            "history_prior_survival_negative_edges": int(prior_negative),
            "max_prior_survival_negative_edges_per_subject": int(max_prior_negative),
            "history_no_prior_continuation_scored_edges": int(no_prior_scored),
            "history_no_prior_continuation_active_edges": int(no_prior_active),
            "history_no_prior_continuation_negative_edges": int(no_prior_negative),
            "max_no_prior_continuation_positive_edges_per_subject": int(max_no_prior_positive),
            "max_no_prior_continuation_abs_weighted_score_per_subject": float(max_no_prior_abs),
            "history_growth_prediction_evaluated_edges": int(growth_evaluated),
            "history_growth_prediction_penalized_edges": int(growth_penalized),
            "max_growth_prediction_penalized_edges_per_subject": int(max_growth_penalized),
            "max_growth_prediction_weighted_penalty_per_subject": float(max_growth_weighted),
            "limit_max_prior_survival_negative_edges_per_subject": int(cfg.max_prior_survival_negative_edges_per_subject),
            "limit_history_prior_survival_negative_edges": int(cfg.max_total_prior_survival_negative_edges),
            "limit_max_no_prior_continuation_positive_edges_per_subject": int(cfg.max_no_prior_continuation_positive_edges_per_subject),
            "limit_history_no_prior_continuation_negative_edges": int(cfg.max_total_no_prior_continuation_negative_edges),
            "limit_max_no_prior_continuation_abs_weighted_score_per_subject": float(cfg.max_no_prior_continuation_abs_weighted_score_per_subject),
            "limit_max_growth_prediction_penalized_edges_per_subject": int(cfg.max_growth_prediction_penalized_edges_per_subject),
            "limit_history_growth_prediction_penalized_edges": int(cfg.max_total_growth_prediction_penalized_edges),
            "limit_max_growth_prediction_weighted_penalty_per_subject": float(cfg.max_growth_prediction_weighted_penalty_per_subject),
        }
    )
    return output


def format_identity_history_promotion_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact identity-history promotion-gate note."""

    sensitivity = dict(decision.get("sensitivity", {}))
    exposure = dict(decision.get("exposure", {}))
    lines = [
        "# FullMHT Identity-History Promotion Gate",
        "",
        f"Status: `{decision.get('status', '')}`",
        f"MHT-vs-local result: `{decision.get('mht_vs_local_result', decision.get('history_search_result', ''))}`",
        f"History-search result: `{decision.get('history_search_result', '')}`",
        f"Prior-control result: `{decision.get('prior_control_result', '')}`",
        f"Track2p-control result: `{decision.get('track2p_control_result', '')}`",
        f"No-local-context result: `{decision.get('no_local_context_control_result', '')}`",
        f"Layer-combination result: `{decision.get('layer_combination_result', '')}`",
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
        "| passing variants per axis | {value} | {limit} |".format(
            value=json.dumps(sensitivity.get("axis_passing_counts", {}), sort_keys=True),
            limit=sensitivity.get("n_required_passing_axis_variants", ""),
        ),
        "",
        "| exposure metric | value | limit |",
        "| --- | ---: | ---: |",
    ]
    for metric, limit_key in (
        ("max_selected_non_prior_edges_per_subject", "limit_max_selected_non_prior_edges_per_subject"),
        ("history_selected_non_prior_edges", "limit_history_selected_non_prior_edges"),
        ("history_switched_prior_successors", "limit_history_switched_prior_successors"),
        ("history_no_prior_successor_continuations", "limit_history_no_prior_successor_continuations"),
        ("history_prior_survival_negative_edges", "limit_history_prior_survival_negative_edges"),
        ("max_prior_survival_negative_edges_per_subject", "limit_max_prior_survival_negative_edges_per_subject"),
        ("history_no_prior_continuation_negative_edges", "limit_history_no_prior_continuation_negative_edges"),
        ("max_no_prior_continuation_positive_edges_per_subject", "limit_max_no_prior_continuation_positive_edges_per_subject"),
        ("history_growth_prediction_penalized_edges", "limit_history_growth_prediction_penalized_edges"),
        ("max_growth_prediction_penalized_edges_per_subject", "limit_max_growth_prediction_penalized_edges_per_subject"),
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
                variants=", ".join(str(item) for item in sensitivity.get("passing_variants", ())) or "none"
            ),
            f"Weak sensitivity axes: {', '.join(str(item) for item in sensitivity.get('weak_axes', ())) or 'none'}",
            f"Pairwise-collapse variants: {', '.join(str(item) for item in sensitivity.get('pairwise_collapse_variants', ())) or 'none'}",
            "Active prior-survival edges: {value}".format(value=exposure.get("history_prior_survival_active_edges", "")),
            "Active no-prior continuation edges: {value}".format(value=exposure.get("history_no_prior_continuation_active_edges", "")),
            "Growth-history penalized edges: {value}".format(value=exposure.get("history_growth_prediction_penalized_edges", "")),
            f"Failed exposure limits: {failed or 'none'}",
        ]
    )
    if exposure.get("missing_columns"):
        missing = ", ".join(str(item) for item in exposure.get("missing_columns", ()))
        lines.append(f"Missing exposure columns: {missing}")
    if exposure.get("inactive_layers"):
        inactive = ", ".join(str(item) for item in exposure.get("inactive_layers", ()))
        lines.append(f"Inactive layers: {inactive}")
    return "\n".join(lines)


def write_identity_history_promotion(decision: Mapping[str, Any], output: Path, *, output_format: str) -> None:
    """Write the promotion gate as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(format_identity_history_promotion_markdown(decision) + "\n", encoding="utf-8")


def _rows_by_approach(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_approach: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        approach = str(row.get("approach", ""))
        if approach:
            by_approach[approach] = row
    return by_approach


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    for item in items:
        if item not in output:
            output.append(item)
    return tuple(output)


def _passing_axis_count(axis_rows: Sequence[str], passing: Sequence[str]) -> int:
    passing_set = set(passing)
    return int(sum(1 for row in axis_rows if row in passing_set))


def _all_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("subject", "")) == "ALL":
            return row
    return None


def _metric(row: Mapping[str, Any], metric: str) -> float:
    try:
        return float(row[metric])
    except KeyError as exc:
        raise ValueError(f"Comparison row is missing metric column {metric!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Comparison metric {metric!r} is not numeric: {row.get(metric)!r}") from exc


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
        prog="python -m bayescatrack.experiments.full_mht_identity_history_promotion_gate",
        description="Combine FullMHT identity-history manifest, sensitivity, and exposure gates.",
    )
    parser.add_argument("canonical_comparison_csv", type=Path)
    parser.add_argument("sensitivity_comparison_csv", type=Path)
    parser.add_argument("exposure_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--min-passing-variants", type=int, default=5)
    parser.add_argument("--min-passing-axis-variants", type=int, default=2)
    parser.add_argument("--max-pairwise-drop", type=float, default=0.01)
    parser.add_argument("--max-non-prior-per-subject", type=int, default=3)
    parser.add_argument("--max-total-non-prior", type=int, default=10)
    parser.add_argument("--max-switches", type=int, default=0)
    parser.add_argument("--max-no-prior-continuations", type=int, default=10)
    parser.add_argument("--max-gap-reactivations", type=int, default=None)
    parser.add_argument("--max-prior-survival-negative-per-subject", type=int, default=3)
    parser.add_argument("--max-total-prior-survival-negative", type=int, default=10)
    parser.add_argument("--max-no-prior-continuation-positive-per-subject", type=int, default=3)
    parser.add_argument("--max-total-no-prior-continuation-negative", type=int, default=10)
    parser.add_argument("--max-no-prior-continuation-abs-weighted-per-subject", type=float, default=8.0)
    parser.add_argument("--max-growth-prediction-penalized-per-subject", type=int, default=3)
    parser.add_argument("--max-total-growth-prediction-penalized", type=int, default=10)
    parser.add_argument("--max-growth-prediction-weighted-penalty-per-subject", type=float, default=8.0)
    parser.add_argument("--require-active-layer-signals", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = IdentityHistoryPromotionConfig(
        sensitivity=IdentityHistorySensitivityConfig(
            min_passing_variants=max(0, int(args.min_passing_variants)),
            min_passing_axis_variants=max(0, int(args.min_passing_axis_variants)),
            max_pairwise_drop=max(0.0, float(args.max_pairwise_drop)),
        ),
        max_selected_non_prior_edges_per_subject=max(0, int(args.max_non_prior_per_subject)),
        max_total_non_prior_edges=max(0, int(args.max_total_non_prior)),
        max_switched_prior_successors=max(0, int(args.max_switches)),
        max_no_prior_successor_continuations=max(0, int(args.max_no_prior_continuations)),
        max_gap_reactivated_tracks=None if args.max_gap_reactivations is None else max(0, int(args.max_gap_reactivations)),
        max_prior_survival_negative_edges_per_subject=max(0, int(args.max_prior_survival_negative_per_subject)),
        max_total_prior_survival_negative_edges=max(0, int(args.max_total_prior_survival_negative)),
        max_no_prior_continuation_positive_edges_per_subject=max(0, int(args.max_no_prior_continuation_positive_per_subject)),
        max_total_no_prior_continuation_negative_edges=max(0, int(args.max_total_no_prior_continuation_negative)),
        max_no_prior_continuation_abs_weighted_score_per_subject=max(0.0, float(args.max_no_prior_continuation_abs_weighted_per_subject)),
        max_growth_prediction_penalized_edges_per_subject=max(0, int(args.max_growth_prediction_penalized_per_subject)),
        max_total_growth_prediction_penalized_edges=max(0, int(args.max_total_growth_prediction_penalized)),
        max_growth_prediction_weighted_penalty_per_subject=max(0.0, float(args.max_growth_prediction_weighted_penalty_per_subject)),
        require_active_layer_signals=bool(args.require_active_layer_signals),
    )
    decision = evaluate_identity_history_promotion(
        load_comparison_rows(args.canonical_comparison_csv),
        load_comparison_rows(args.sensitivity_comparison_csv),
        load_exposure_rows(args.exposure_csv),
        config=cfg,
    )
    if args.output is not None:
        write_identity_history_promotion(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_identity_history_promotion_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
