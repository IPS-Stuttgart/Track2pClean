"""Seed-session sensitivity audit for Track2p-policy component cleanup.

The official Track2p benchmark is seed-anchored: predictions and references are
filtered through one seed-session ROI set before pairwise and complete-track
scoring.  This diagnostic holds the ComponentCleanup method fixed and varies
only the seed session to distinguish edge evidence failures from seed-anchoring
protocol effects.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
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
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    _evaluated_prediction_rows,
    _mark_applied_splits,
    _normalize_int_track_matrix,
    apply_weakest_bridge_splits,
    component_audit_rows,
)
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    _no_prune_config,
    residual_error_rows,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_SEED_SENSITIVITY_AUDIT_METHOD = (
    "track2p-policy-seed-sensitivity-audit"
)
CompleteTrack = tuple[int, ...]


@dataclass(frozen=True)
class SeedSensitivityAuditResult:
    """Aggregate seed rows plus per-reference-track recoverability rows."""

    summary_rows: tuple[dict[str, float | int | str], ...]
    track_rows: tuple[dict[str, float | int | str], ...]


def run_track2p_policy_seed_sensitivity_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    seed_sessions: Sequence[int] | str = "all",
    cleanup_config: ComponentCleanupConfig | None = None,
) -> SeedSensitivityAuditResult:
    """Run ComponentCleanup once per seed session and label recoverable GT rows."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    cleanup_config = cleanup_config or ComponentCleanupConfig()
    all_summary_rows: list[dict[str, float | int | str]] = []
    all_track_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy seed-sensitivity audit requires independent "
                "manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        policy_prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=_no_prune_config(),
        )
        policy_full = _normalize_int_track_matrix(policy_prediction.tracks)
        resolved_seeds = _resolved_seed_sessions(
            seed_sessions, n_sessions=reference_tracks.shape[1]
        )
        subject_summary_rows: list[dict[str, float | int | str]] = []
        subject_track_rows: list[dict[str, float | int | str]] = []
        for seed_session in resolved_seeds:
            seed_config = replace(policy_config, seed_session=int(seed_session))
            predicted_eval, reference_eval, _policy_eval = (
                _component_cleanup_eval_from_prediction(
                    policy_full,
                    policy_prediction.diagnostics,
                    sessions=sessions,
                    reference_tracks=reference_tracks,
                    config=seed_config,
                    cleanup_config=cleanup_config,
                    threshold_method=threshold_method,
                    iou_distance_threshold=float(iou_distance_threshold),
                    subject=subject_dir.name,
                )
            )
            residual_rows = residual_error_rows(
                predicted_eval,
                reference_eval,
                subject=subject_dir.name,
                sessions=sessions,
                seed_session=int(seed_session),
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                cell_probability_threshold=float(
                    seed_config.cell_probability_threshold
                ),
                transform_type=str(seed_config.transform_type),
            )
            scores = _score_counts(predicted_eval, reference_eval)
            missing_seed_complete_fns = _missing_seed_complete_fns(residual_rows)
            track_rows = _complete_reference_track_rows(
                subject=subject_dir.name,
                seed_session=int(seed_session),
                predicted=predicted_eval,
                reference=reference_eval,
                residual_rows=residual_rows,
            )
            subject_track_rows.extend(track_rows)
            subject_summary_rows.append(
                {
                    "subject": subject_dir.name,
                    "seed_session": int(seed_session),
                    "pairwise_true_positives": int(
                        scores["pairwise_true_positives"]
                    ),
                    "pairwise_false_positives": int(
                        scores["pairwise_false_positives"]
                    ),
                    "pairwise_false_negatives": int(
                        scores["pairwise_false_negatives"]
                    ),
                    "pairwise_f1_micro": float(scores["pairwise_f1"]),
                    "complete_track_true_positives": int(
                        scores["complete_track_true_positives"]
                    ),
                    "complete_track_false_positives": int(
                        scores["complete_track_false_positives"]
                    ),
                    "complete_track_false_negatives": int(
                        scores["complete_track_false_negatives"]
                    ),
                    "complete_track_f1_micro": float(scores["complete_track_f1"]),
                    "missing_seed_complete_fns": int(missing_seed_complete_fns),
                    "complete_fns_recovered_under_other_seed": 0,
                    "missing_seed_complete_fns_recovered_under_other_seed": 0,
                    "reference_complete_tracks": int(
                        scores["reference_complete_tracks"]
                    ),
                    "evaluated_prediction_tracks": int(predicted_eval.shape[0]),
                    "threshold_method": str(threshold_method),
                    "iou_distance_threshold": float(iou_distance_threshold),
                    "cell_probability_threshold": float(
                        seed_config.cell_probability_threshold
                    ),
                    "transform_type": str(seed_config.transform_type),
                }
            )
        subject_track_rows = _annotate_recoverability(subject_track_rows)
        subject_summary_rows = _annotate_summary_recoverability(
            subject_summary_rows, subject_track_rows
        )
        all_summary_rows.extend(subject_summary_rows)
        all_track_rows.extend(subject_track_rows)
    all_summary_rows.extend(_aggregate_seed_rows(all_summary_rows))
    return SeedSensitivityAuditResult(
        summary_rows=tuple(all_summary_rows),
        track_rows=tuple(all_track_rows),
    )


def _component_cleanup_eval_from_prediction(
    policy_full: np.ndarray,
    diagnostics: Sequence[Track2pPolicyLinkDiagnostic],
    *,
    sessions: Sequence[Any],
    reference_tracks: np.ndarray,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    subject: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy_eval, reference_eval, evaluated_track_ids = _evaluated_prediction_rows(
        policy_full,
        reference_tracks,
        config=config,
    )
    audit_rows = component_audit_rows(
        policy_eval,
        reference_eval,
        sessions=sessions,
        diagnostics=diagnostics,
        subject=subject,
        config=cleanup_config,
        track_ids=evaluated_track_ids,
        seed_session=config.seed_session,
    )
    cleaned_full = apply_weakest_bridge_splits(
        policy_full, _mark_applied_splits(audit_rows, apply_splits=True)
    )
    cleaned_eval, reference_eval, _ = _evaluated_prediction_rows(
        cleaned_full, reference_tracks, config=config
    )
    return cleaned_eval, reference_eval, policy_eval


def _score_counts(
    predicted_eval: np.ndarray, reference_eval: np.ndarray
) -> dict[str, float | int]:
    scores = dict(score_track_matrices(predicted_eval, reference_eval))
    for prefix in ("pairwise", "complete_track"):
        scores[f"{prefix}_f1"] = _f1_from_counts(
            int(scores[f"{prefix}_true_positives"]),
            int(scores[f"{prefix}_false_positives"]),
            int(scores[f"{prefix}_false_negatives"]),
        )
    scores["reference_complete_tracks"] = int(
        sum(_complete_track_counter(reference_eval).values())
    )
    return scores


def _complete_reference_track_rows(
    *,
    subject: str,
    seed_session: int,
    predicted: np.ndarray,
    reference: np.ndarray,
    residual_rows: Sequence[Mapping[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    predicted_counts = _complete_track_counter(predicted)
    reference_counts = _complete_track_counter(reference)
    residual_by_track = {
        (str(row["track_id_or_edge"]), int(row["occurrence_index"])): row
        for row in residual_rows
        if str(row["error_type"]) == "complete_fn"
    }
    rows: list[dict[str, float | int | str]] = []
    for track in sorted(reference_counts):
        track_id = _track_id(track)
        for occurrence_index in range(int(reference_counts[track])):
            true_positive = predicted_counts.get(track, 0) > occurrence_index
            residual = residual_by_track.get((track_id, occurrence_index), {})
            reason = "" if true_positive else str(residual.get("reason_bucket", ""))
            rows.append(
                {
                    "subject": subject,
                    "seed_session": int(seed_session),
                    "reference_track_id": track_id,
                    "occurrence_index": int(occurrence_index),
                    "status": "true_positive" if true_positive else "false_negative",
                    "reason_bucket": reason,
                    "missing_seed_complete_fn": int(
                        (not true_positive) and reason == "missing seed-session ROI"
                    ),
                    "recoverable_under_other_seed": 0,
                    "recovered_seed_sessions": "",
                }
            )
    return rows


def _annotate_recoverability(
    rows: Sequence[Mapping[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    true_positive_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        if str(row["status"]) == "true_positive":
            true_positive_seeds[
                (str(row["subject"]), str(row["reference_track_id"]))
            ].add(int(row["seed_session"]))
    output: list[dict[str, float | int | str]] = []
    for row in rows:
        updated = dict(row)
        seeds = sorted(
            seed
            for seed in true_positive_seeds.get(
                (str(row["subject"]), str(row["reference_track_id"])), set()
            )
            if seed != int(row["seed_session"])
        )
        if str(row["status"]) == "false_negative" and seeds:
            updated["recoverable_under_other_seed"] = 1
            updated["recovered_seed_sessions"] = ",".join(str(seed) for seed in seeds)
        output.append(updated)
    return output


def _annotate_summary_recoverability(
    summary_rows: Sequence[Mapping[str, float | int | str]],
    track_rows: Sequence[Mapping[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    recovered_by_seed = Counter(
        int(row["seed_session"])
        for row in track_rows
        if str(row["status"]) == "false_negative"
        and int(row["recoverable_under_other_seed"]) == 1
    )
    missing_seed_recovered_by_seed = Counter(
        int(row["seed_session"])
        for row in track_rows
        if str(row["status"]) == "false_negative"
        and int(row["missing_seed_complete_fn"]) == 1
        and int(row["recoverable_under_other_seed"]) == 1
    )
    output: list[dict[str, float | int | str]] = []
    for row in summary_rows:
        seed_session = int(row["seed_session"])
        updated = dict(row)
        updated["complete_fns_recovered_under_other_seed"] = int(
            recovered_by_seed[seed_session]
        )
        updated["missing_seed_complete_fns_recovered_under_other_seed"] = int(
            missing_seed_recovered_by_seed[seed_session]
        )
        output.append(updated)
    return output


def _aggregate_seed_rows(
    rows: Sequence[Mapping[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    by_seed: dict[int, list[Mapping[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        if str(row["subject"]) == "ALL":
            continue
        by_seed[int(row["seed_session"])].append(row)
    output: list[dict[str, float | int | str]] = []
    for seed_session, seed_rows in sorted(by_seed.items()):
        pairwise_tp = _sum_int(seed_rows, "pairwise_true_positives")
        pairwise_fp = _sum_int(seed_rows, "pairwise_false_positives")
        pairwise_fn = _sum_int(seed_rows, "pairwise_false_negatives")
        complete_tp = _sum_int(seed_rows, "complete_track_true_positives")
        complete_fp = _sum_int(seed_rows, "complete_track_false_positives")
        complete_fn = _sum_int(seed_rows, "complete_track_false_negatives")
        output.append(
            {
                "subject": "ALL",
                "seed_session": int(seed_session),
                "pairwise_true_positives": pairwise_tp,
                "pairwise_false_positives": pairwise_fp,
                "pairwise_false_negatives": pairwise_fn,
                "pairwise_f1_micro": _f1_from_counts(
                    pairwise_tp, pairwise_fp, pairwise_fn
                ),
                "complete_track_true_positives": complete_tp,
                "complete_track_false_positives": complete_fp,
                "complete_track_false_negatives": complete_fn,
                "complete_track_f1_micro": _f1_from_counts(
                    complete_tp, complete_fp, complete_fn
                ),
                "missing_seed_complete_fns": _sum_int(
                    seed_rows, "missing_seed_complete_fns"
                ),
                "complete_fns_recovered_under_other_seed": _sum_int(
                    seed_rows, "complete_fns_recovered_under_other_seed"
                ),
                "missing_seed_complete_fns_recovered_under_other_seed": _sum_int(
                    seed_rows,
                    "missing_seed_complete_fns_recovered_under_other_seed",
                ),
                "reference_complete_tracks": _sum_int(
                    seed_rows, "reference_complete_tracks"
                ),
                "evaluated_prediction_tracks": _sum_int(
                    seed_rows, "evaluated_prediction_tracks"
                ),
                "threshold_method": str(seed_rows[0]["threshold_method"]),
                "iou_distance_threshold": float(
                    seed_rows[0]["iou_distance_threshold"]
                ),
                "cell_probability_threshold": float(
                    seed_rows[0]["cell_probability_threshold"]
                ),
                "transform_type": str(seed_rows[0]["transform_type"]),
            }
        )
    return output


def _missing_seed_complete_fns(
    rows: Sequence[Mapping[str, float | int | str]],
) -> int:
    return int(
        sum(
            1
            for row in rows
            if str(row.get("error_type")) == "complete_fn"
            and str(row.get("reason_bucket")) == "missing seed-session ROI"
        )
    )


def _complete_track_counter(track_matrix: np.ndarray) -> Counter[CompleteTrack]:
    counter: Counter[CompleteTrack] = Counter()
    for row in np.asarray(track_matrix, dtype=int):
        if np.all(row >= 0):
            counter[tuple(int(value) for value in row)] += 1
    return counter


def _resolved_seed_sessions(
    seed_sessions: Sequence[int] | str, *, n_sessions: int
) -> tuple[int, ...]:
    if isinstance(seed_sessions, str):
        if seed_sessions.casefold() == "all":
            return tuple(range(int(n_sessions)))
        values = tuple(
            int(value.strip())
            for value in seed_sessions.split(",")
            if value.strip()
        )
    else:
        values = tuple(int(value) for value in seed_sessions)
    if not values:
        raise ValueError("seed_sessions must not be empty")
    for seed_session in values:
        if seed_session < 0 or seed_session >= int(n_sessions):
            raise ValueError(
                f"seed_session {seed_session} out of bounds for {n_sessions} sessions"
            )
    return values


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)
    if denominator == 0:
        return 1.0
    return float(2 * int(tp) / denominator)


def _sum_int(rows: Sequence[Mapping[str, float | int | str]], key: str) -> int:
    return int(sum(int(row[key]) for row in rows))


def _track_id(track: CompleteTrack) -> str:
    return ",".join(str(value) for value in track)


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write audit rows as CSV or JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for seed-sensitivity audit."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-seed-sensitivity-audit",
        description=(
            "Vary only the seed session for Track2p-policy ComponentCleanup "
            "and report complete-track recoverability."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        default="manual-gt",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", choices=("auto", "suite2p", "npy"), default="suite2p"
    )
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
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
    parser.add_argument(
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--seed-sessions",
        default="all",
        help="Comma-separated seed sessions, or 'all'.",
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-output", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--track-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy seed-sensitivity audit CLI."""

    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        plane_name=args.plane_name,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    result = run_track2p_policy_seed_sensitivity_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        seed_sessions=str(args.seed_sessions),
        cleanup_config=cleanup_config,
    )
    write_rows(result.summary_rows, args.output, output_format=args.format)
    if args.track_output is not None:
        write_rows(
            result.track_rows,
            args.track_output,
            output_format=cast(Literal["csv", "json"], args.track_format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
