"""Label-free exposure audit for FullMHT on Track2p-style subjects.

Benchmark rows need manual-GT references for scoring.  Exposure audits should not:
they answer whether an opt-in FullMHT method layer fires broadly on all available
Track2p-style subjects.  This module runs the same scan-assignment beam from
Track2p seed/proposal tracks and writes per-subject behavior counts without
loading manual-GT labels or benchmark audit columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from bayescatrack.experiments import (
    track2p_policy_suffix_stitch_ranking_audit as rank,
)
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _load_subject_sessions,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_full_mht_benchmark import (
    FullMHTConfig,
    _MHTHypothesis,
    _active_track_sources,
    _advance_scan,
    _prune_output_tracks,
    _seed_rois,
    _select_final_hypothesis,
    _track2p_prediction_for_subject,
    _track_edges,
)

METHOD = "track2p-policy-full-mht-exposure-audit"


@dataclass(frozen=True)
class FullMHTExposureAuditResult:
    """Per-subject exposure rows for a label-free FullMHT audit."""

    rows: tuple[dict[str, Any], ...]


def run_full_mht_exposure_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    mht_config: FullMHTConfig | None = None,
    progress: bool = False,
) -> FullMHTExposureAuditResult:
    """Run a label-free FullMHT exposure audit on all discovered subjects."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    mht_config = mht_config or FullMHTConfig(seed_source="track2p-output")
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    rows: list[dict[str, Any]] = []
    for subject_index, subject_dir in enumerate(subject_dirs, start=1):
        if progress:
            print(
                f"{METHOD}: subject {subject_index}/{len(subject_dirs)} "
                f"{subject_dir.name}",
                flush=True,
            )
        rows.append(
            _run_subject_exposure_audit(
                subject_dir,
                config=policy_config,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                mht_config=mht_config,
                progress=progress,
            )
        )
    rows.append(_all_subjects_row(rows))
    return FullMHTExposureAuditResult(tuple(rows))


def _run_subject_exposure_audit(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    mht_config: FullMHTConfig,
    progress: bool,
) -> dict[str, Any]:
    sessions = _load_subject_sessions(subject_dir, config)
    n_sessions = len(sessions)
    if n_sessions < 2:
        raise ValueError(f"{subject_dir.name} has fewer than two sessions")

    track2p_prediction = _track2p_prediction_for_subject(subject_dir, config=config)
    seed_rois = _seed_rois(
        sessions,
        np.zeros((0, n_sessions), dtype=int),
        seed_session=int(config.seed_session),
        seed_source="track2p-output",
        cell_probability_threshold=float(config.cell_probability_threshold),
        track2p_tracks=track2p_prediction,
    )
    if mht_config.max_seed_tracks is not None:
        seed_rois = seed_rois[: max(0, int(mht_config.max_seed_tracks))]
    if not seed_rois:
        return _empty_subject_row(subject_dir.name, n_sessions=n_sessions)

    initial = np.full((len(seed_rois), n_sessions), -1, dtype=int)
    initial[:, int(config.seed_session)] = np.asarray(seed_rois, dtype=int)
    hypotheses: list[_MHTHypothesis] = [_MHTHypothesis(initial, 0.0, tuple())]
    track2p_prior_edges = _track_edges(np.asarray(track2p_prediction, dtype=int))
    feature_cache = rank._FeatureCache(
        sessions=sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(config.cell_probability_threshold),
        matrices={},
    )

    for session_index in range(int(config.seed_session), n_sessions - 1):
        if progress:
            active_count = sum(
                len(
                    _active_track_sources(
                        hypothesis.tracks,
                        session_index=int(session_index),
                        max_gap=int(mht_config.max_gap),
                    )
                )
                for hypothesis in hypotheses
            )
            print(
                f"{METHOD}: {subject_dir.name} scan "
                f"{session_index}->{session_index + 1} "
                f"hypotheses={len(hypotheses)} active_sources={active_count}",
                flush=True,
            )
        hypotheses = _advance_scan(
            hypotheses,
            sessions=sessions,
            feature_cache=feature_cache,
            session_index=int(session_index),
            config=mht_config,
            track2p_prior_edges=track2p_prior_edges,
        )

    best, final_selection = _select_final_hypothesis(
        hypotheses,
        sessions=sessions,
        feature_cache=feature_cache,
        config=mht_config,
        track2p_prior_edges=track2p_prior_edges,
    )
    output_tracks = _prune_output_tracks(
        best.tracks,
        min_observations=int(mht_config.min_output_observations),
    )
    selected_edges = _track_edges(np.asarray(output_tracks, dtype=int))
    selected_prior_edges = selected_edges & track2p_prior_edges
    selected_non_prior_edges = selected_edges - track2p_prior_edges
    history_totals = _history_totals(best.history)
    return {
        "subject": subject_dir.name,
        "n_sessions": int(n_sessions),
        "n_seed_tracks": int(len(seed_rois)),
        "final_hypotheses": int(len(hypotheses)),
        "selected_rank": int(final_selection.get("terminal_selected_rank", 1)),
        "best_score": float(best.score),
        "terminal_adjusted_score": float(
            final_selection.get("terminal_adjusted_score", best.score)
        ),
        "terminal_history_risk": float(
            final_selection.get("terminal_history_risk", 0.0)
        ),
        "terminal_identity_history_risk": float(
            final_selection.get("terminal_identity_history_risk", 0.0)
        ),
        "terminal_motion_history_risk": float(
            final_selection.get("terminal_motion_history_risk", 0.0)
        ),
        "n_output_tracks": int(output_tracks.shape[0]),
        "n_selected_edges": int(len(selected_edges)),
        "n_selected_prior_edges": int(len(selected_prior_edges)),
        "n_selected_non_prior_edges": int(len(selected_non_prior_edges)),
        "n_missing_observations": int(_missing_observation_count(output_tracks)),
        **history_totals,
    }


def _history_totals(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "assigned_edges",
        "missed_tracks",
        "selected_prior_edges",
        "selected_non_prior_edges",
        "missed_prior_successors",
        "switched_prior_successors",
        "no_prior_successor_continuations",
        "gap_reactivated_tracks",
        "scan_candidates",
    )
    totals: dict[str, Any] = {
        f"history_{key}": _history_int_sum(history, key) for key in keys
    }
    totals.update(_history_growth_prediction_totals(history))
    return totals


def _history_int_sum(history: Sequence[Mapping[str, Any]], key: str) -> int:
    total = 0
    for item in history:
        try:
            total += int(item.get(key, 0))
        except (TypeError, ValueError):
            continue
    return int(total)


def _history_growth_prediction_totals(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evaluated = 0
    penalized = 0
    penalty_sum = 0.0
    weighted_sum = 0.0
    for item in history:
        summaries = str(item.get("selected_edge_summaries", ""))
        if not summaries:
            continue
        for summary in summaries.split(";"):
            values = _summary_key_values(summary)
            if "growth_pred" not in values and "growth_pred_weighted" not in values:
                continue
            evaluated += 1
            penalty = _summary_float(values.get("growth_pred"), 0.0)
            weighted = _summary_float(values.get("growth_pred_weighted"), 0.0)
            if penalty > 0.0 or weighted > 0.0:
                penalized += 1
            penalty_sum += max(0.0, penalty)
            weighted_sum += max(0.0, weighted)
    return {
        "history_growth_prediction_evaluated_edges": int(evaluated),
        "history_growth_prediction_penalized_edges": int(penalized),
        "history_growth_prediction_penalty": float(penalty_sum),
        "history_growth_prediction_weighted_penalty": float(weighted_sum),
    }


def _summary_key_values(summary: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in str(summary).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _summary_float(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return numeric if np.isfinite(numeric) else float(fallback)


def _missing_observation_count(matrix: np.ndarray) -> int:
    tracks = np.asarray(matrix, dtype=int)
    if tracks.ndim != 2 or tracks.size == 0:
        return 0
    observed_rows = np.any(tracks >= 0, axis=1)
    if not np.any(observed_rows):
        return 0
    return int(np.sum((tracks[observed_rows] < 0).astype(int)))


def _empty_subject_row(subject: str, *, n_sessions: int) -> dict[str, Any]:
    return {
        "subject": subject,
        "n_sessions": int(n_sessions),
        "n_seed_tracks": 0,
        "final_hypotheses": 0,
        "selected_rank": 0,
        "best_score": 0.0,
        "terminal_adjusted_score": 0.0,
        "terminal_history_risk": 0.0,
        "terminal_identity_history_risk": 0.0,
        "terminal_motion_history_risk": 0.0,
        "n_output_tracks": 0,
        "n_selected_edges": 0,
        "n_selected_prior_edges": 0,
        "n_selected_non_prior_edges": 0,
        "n_missing_observations": 0,
        **_history_totals(tuple()),
    }


def _all_subjects_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "n_sessions",
        "n_seed_tracks",
        "final_hypotheses",
        "n_output_tracks",
        "n_selected_edges",
        "n_selected_prior_edges",
        "n_selected_non_prior_edges",
        "n_missing_observations",
        "history_assigned_edges",
        "history_missed_tracks",
        "history_selected_prior_edges",
        "history_selected_non_prior_edges",
        "history_missed_prior_successors",
        "history_switched_prior_successors",
        "history_no_prior_successor_continuations",
        "history_gap_reactivated_tracks",
        "history_scan_candidates",
        "history_growth_prediction_evaluated_edges",
        "history_growth_prediction_penalized_edges",
    ]
    float_keys = [
        "history_growth_prediction_penalty",
        "history_growth_prediction_weighted_penalty",
    ]
    output: dict[str, Any] = {"subject": "ALL"}
    for key in numeric_keys:
        output[key] = int(sum(int(row.get(key, 0)) for row in rows))
    for key in float_keys:
        output[key] = float(sum(float(row.get(key, 0.0)) for row in rows))
    output["max_selected_non_prior_edges_per_subject"] = max(
        (int(row.get("n_selected_non_prior_edges", 0)) for row in rows),
        default=0,
    )
    output["max_missing_observations_per_subject"] = max(
        (int(row.get("n_missing_observations", 0)) for row in rows),
        default=0,
    )
    output["max_growth_prediction_penalized_edges_per_subject"] = max(
        (int(row.get("history_growth_prediction_penalized_edges", 0)) for row in rows),
        default=0,
    )
    output["max_growth_prediction_weighted_penalty_per_subject"] = max(
        (
            float(row.get("history_growth_prediction_weighted_penalty", 0.0))
            for row in rows
        ),
        default=0.0,
    )
    return output


def _write_rows(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    output_format: Literal["csv", "json"],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
        return
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "bayescatrack.experiments.track2p_policy_full_mht_exposure_audit"
        ),
        description="Run a label-free FullMHT exposure audit.",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format",
        choices=("auto", "suite2p", "npy"),
        default="suite2p",
    )
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    parser.add_argument("--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE)
    parser.add_argument(
        "--iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--scan-hypotheses", type=int, default=8)
    parser.add_argument("--edge-top-k", type=int, default=4)
    parser.add_argument(
        "--identity-diverse-beam",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--miss-cost", type=float, default=2.0)
    parser.add_argument("--full-mht-max-gap", type=int, default=1)
    parser.add_argument("--gap-reactivation-cost", type=float, default=1.0)
    parser.add_argument("--min-output-observations", type=int, default=1)
    parser.add_argument("--min-edge-score", type=float, default=0.25)
    parser.add_argument("--max-seed-tracks", type=int, default=None)
    parser.add_argument("--track2p-prior-weight", type=float, default=12.0)
    parser.add_argument("--track2p-non-prior-penalty", type=float, default=2.0)
    parser.add_argument("--track2p-prior-switch-penalty", type=float, default=8.0)
    parser.add_argument(
        "--track2p-no-prior-successor-penalty",
        type=float,
        default=8.0,
    )
    parser.add_argument("--track2p-prior-miss-penalty", type=float, default=4.0)
    parser.add_argument(
        "--terminal-incomplete-history-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--terminal-motion-history-weight", type=float, default=0.0)
    parser.add_argument("--growth-history-prediction-weight", type=float, default=0.0)
    parser.add_argument("--growth-history-prediction-scale", type=float, default=1.0)
    parser.add_argument("--growth-history-prediction-clip", type=float, default=8.0)
    parser.add_argument("--growth-history-prediction-min-edges", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if float(args.terminal_incomplete_history_weight) > 0.0:
        from bayescatrack.experiments.full_mht_terminal_completion_integration import (
            install_full_mht_terminal_completion_objective,
        )

        install_full_mht_terminal_completion_objective()
    if float(args.terminal_motion_history_weight) > 0.0:
        from bayescatrack.experiments.full_mht_history_dynamics_integration import (
            install_full_mht_history_dynamics_objective,
        )

        install_full_mht_history_dynamics_objective()
    if float(args.growth_history_prediction_weight) > 0.0:
        from bayescatrack.experiments.full_mht_growth_history_prediction_integration import (
            install_full_mht_growth_history_prediction_scoring,
        )

        install_full_mht_growth_history_prediction_scoring()

    benchmark_config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=None,
        reference_kind="auto",
        plane_name=args.plane_name,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=False,
        transform_type=args.transform_type,
        allow_track2p_as_reference_for_smoke_test=True,
        include_behavior=False,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    mht_config = FullMHTConfig(
        beam_width=max(1, int(args.beam_width)),
        scan_hypotheses=max(1, int(args.scan_hypotheses)),
        edge_top_k=max(1, int(args.edge_top_k)),
        identity_diverse_beam=bool(args.identity_diverse_beam),
        miss_cost=float(args.miss_cost),
        max_gap=max(0, int(args.full_mht_max_gap)),
        gap_reactivation_cost=float(args.gap_reactivation_cost),
        min_output_observations=max(1, int(args.min_output_observations)),
        min_edge_score=float(args.min_edge_score),
        seed_source="track2p-output",
        max_seed_tracks=args.max_seed_tracks,
        track2p_prior_weight=float(args.track2p_prior_weight),
        track2p_non_prior_penalty=float(args.track2p_non_prior_penalty),
        track2p_prior_switch_penalty=float(args.track2p_prior_switch_penalty),
        track2p_no_prior_successor_penalty=float(
            args.track2p_no_prior_successor_penalty
        ),
        track2p_prior_miss_penalty=float(args.track2p_prior_miss_penalty),
    )
    object.__setattr__(
        mht_config,
        "terminal_incomplete_history_weight",
        float(args.terminal_incomplete_history_weight),
    )
    object.__setattr__(
        mht_config,
        "terminal_motion_history_weight",
        float(args.terminal_motion_history_weight),
    )
    object.__setattr__(
        mht_config,
        "growth_history_prediction_weight",
        float(args.growth_history_prediction_weight),
    )
    object.__setattr__(
        mht_config,
        "growth_history_prediction_scale",
        float(args.growth_history_prediction_scale),
    )
    object.__setattr__(
        mht_config,
        "growth_history_prediction_clip",
        float(args.growth_history_prediction_clip),
    )
    object.__setattr__(
        mht_config,
        "growth_history_prediction_min_edges",
        max(1, int(args.growth_history_prediction_min_edges)),
    )
    result = run_full_mht_exposure_audit(
        benchmark_config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        mht_config=mht_config,
        progress=bool(args.progress),
    )
    _write_rows(
        result.rows,
        args.output,
        output_format=cast(Literal["csv", "json"], args.format),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
