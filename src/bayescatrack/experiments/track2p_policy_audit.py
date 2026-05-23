"""Duplicate-aware edge ledger for the Track2p-policy benchmark row.

The Track2p-policy min result is now close to Track2p pairwise F1 and better on
complete-track F1. The next useful workstream is prune-only: identify the small
number of false-positive policy edges without adding rescue edges. This module
exports the per-subject, per-session-pair TP/FP/FN ledger needed for that
analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from bayescatrack.evaluation.track2p_metrics import normalize_track_matrix
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig

OutputFormat = Literal["csv", "json"]
ThresholdMethod = Literal["otsu", "min"]
TrackEdge = tuple[int, int, int, int]

TRACK2P_POLICY_AUDIT_DEFAULT_TRANSFORM_TYPE = "affine"
TRACK2P_POLICY_AUDIT_DEFAULT_THRESHOLD_METHOD: ThresholdMethod = "min"
TRACK2P_POLICY_AUDIT_DEFAULT_IOU_DISTANCE_THRESHOLD = 12.0
TRACK2P_POLICY_AUDIT_DEFAULT_CELL_PROBABILITY_THRESHOLD = 0.5


def pairwise_edge_counter(
    track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> Counter[TrackEdge]:
    """Return duplicate-aware pairwise edge counts from a track matrix."""

    matrix = normalize_track_matrix(track_matrix)
    pairs = _session_pairs(matrix, session_pairs=session_pairs)
    counter: Counter[TrackEdge] = Counter()
    for session_a, session_b in pairs:
        for row in matrix:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if _valid_roi(roi_a) and _valid_roi(roi_b):
                counter[(session_a, session_b, int(roi_a), int(roi_b))] += 1
    return counter


def pairwise_edge_ledger_rows(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    subject: str = "",
    session_names: Sequence[str] | None = None,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> list[dict[str, int | str]]:
    """Return duplicate-aware TP/FP/FN rows for predicted-vs-reference edges."""

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )

    pairs = _session_pairs(predicted, session_pairs=session_pairs)
    predicted_counts = pairwise_edge_counter(predicted, session_pairs=pairs)
    reference_counts = pairwise_edge_counter(reference, session_pairs=pairs)
    names = tuple(session_names or ())
    rows: list[dict[str, int | str]] = []
    for edge in sorted(set(predicted_counts) | set(reference_counts)):
        session_a, session_b, source_roi, target_roi = edge
        predicted_count = int(predicted_counts.get(edge, 0))
        reference_count = int(reference_counts.get(edge, 0))
        true_positive_count = min(predicted_count, reference_count)
        false_positive_count = max(predicted_count - reference_count, 0)
        false_negative_count = max(reference_count - predicted_count, 0)
        rows.append(
            {
                "subject": subject,
                "session_a": int(session_a),
                "session_b": int(session_b),
                "session_a_name": _session_name(names, session_a),
                "session_b_name": _session_name(names, session_b),
                "source_roi": int(source_roi),
                "target_roi": int(target_roi),
                "predicted_count": predicted_count,
                "reference_count": reference_count,
                "true_positive_count": true_positive_count,
                "false_positive_count": false_positive_count,
                "false_negative_count": false_negative_count,
                "classification": _edge_classification(
                    true_positive_count=true_positive_count,
                    false_positive_count=false_positive_count,
                    false_negative_count=false_negative_count,
                ),
            }
        )
    return rows


def run_track2p_policy_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_AUDIT_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_AUDIT_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
) -> list[dict[str, int | float | str]]:
    """Run Track2p-policy and return a per-edge manual-GT ledger."""

    from bayescatrack.experiments.track2p_benchmark import (
        _filter_tracks_by_seed_rois,
        _load_reference_for_subject,
        _load_subject_sessions,
        _reference_matrix,
        _reference_seed_roi_set,
        _score_prediction_against_reference,
        _validate_reference_for_benchmark,
        _validate_reference_roi_indices,
        discover_subject_dirs,
    )
    from bayescatrack.experiments.track2p_emulation_benchmark import (
        emulate_track2p_tracks,
    )
    from bayescatrack.experiments.track2p_policy_benchmark import (
        track2p_policy_config,
    )

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

    rows: list[dict[str, int | float | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        predicted = emulate_track2p_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
        )
        reference_matrix = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        predicted_for_scoring = normalize_track_matrix(predicted)
        reference_for_scoring = reference_matrix
        if policy_config.restrict_to_reference_seed_rois:
            seed_rois = _reference_seed_roi_set(
                reference_for_scoring, seed_session=policy_config.seed_session
            )
            predicted_for_scoring = _filter_tracks_by_seed_rois(
                predicted_for_scoring,
                seed_rois,
                seed_session=policy_config.seed_session,
            )
            reference_for_scoring = _filter_tracks_by_seed_rois(
                reference_for_scoring,
                seed_rois,
                seed_session=policy_config.seed_session,
            )
        scores = _score_prediction_against_reference(
            predicted, reference, config=policy_config
        )
        session_names = tuple(session.session_name for session in sessions)
        subject_rows = pairwise_edge_ledger_rows(
            predicted_for_scoring,
            reference_for_scoring,
            subject=subject_dir.name,
            session_names=session_names,
        )
        for row in subject_rows:
            rows.append(
                {
                    **row,
                    "policy_threshold_method": str(threshold_method),
                    "policy_iou_distance_threshold": float(iou_distance_threshold),
                    "policy_transform_type": str(policy_config.transform_type),
                    "policy_cell_probability_threshold": float(
                        policy_config.cell_probability_threshold
                    ),
                    "subject_pairwise_f1": float(scores["pairwise_f1"]),
                    "subject_complete_track_f1": float(scores["complete_track_f1"]),
                }
            )
    return rows


def write_audit_rows(
    rows: Sequence[Mapping[str, Any]], output: Path, output_format: OutputFormat
) -> None:
    """Write audit rows as CSV or JSON."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
        return
    fieldnames = _audit_fieldnames(rows)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Track2p-policy audit CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-audit",
        description="Export a duplicate-aware edge ledger for Track2p-policy predictions.",
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
        "--transform-type", default=TRACK2P_POLICY_AUDIT_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_AUDIT_DEFAULT_THRESHOLD_METHOD,
    )
    parser.add_argument(
        "--iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_AUDIT_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_AUDIT_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy edge audit CLI."""

    args = build_arg_parser().parse_args(argv)
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        max_gap=1,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=False,
        weighted_centroids=False,
        exclude_overlapping_pixels=False,
    )
    rows = run_track2p_policy_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
    )
    write_audit_rows(rows, args.output, cast(OutputFormat, args.format))
    return 0


def _session_pairs(
    matrix: np.ndarray,
    *,
    session_pairs: Iterable[tuple[int, int]] | None,
) -> tuple[tuple[int, int], ...]:
    pairs = (
        tuple((index, index + 1) for index in range(max(0, matrix.shape[1] - 1)))
        if session_pairs is None
        else tuple((int(a), int(b)) for a, b in session_pairs)
    )
    for session_a, session_b in pairs:
        if session_a < 0 or session_b < 0:
            raise IndexError("session indices must be non-negative")
        if session_a >= matrix.shape[1] or session_b >= matrix.shape[1]:
            raise IndexError(
                f"session pair {(session_a, session_b)!r} out of bounds for {matrix.shape[1]} sessions"
            )
        if session_a >= session_b:
            raise ValueError("session pairs must point forward in time")
    return pairs


def _valid_roi(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    try:
        return int(cast(Any, value)) >= 0
    except (TypeError, ValueError):
        return False


def _session_name(session_names: Sequence[str], session_index: int) -> str:
    if 0 <= session_index < len(session_names):
        return str(session_names[session_index])
    return ""


def _edge_classification(
    *,
    true_positive_count: int,
    false_positive_count: int,
    false_negative_count: int,
) -> str:
    labels: list[str] = []
    if true_positive_count:
        labels.append("true_positive")
    if false_positive_count:
        labels.append("false_positive")
    if false_negative_count:
        labels.append("false_negative")
    return "+".join(labels) if labels else "empty"


def _audit_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "subject",
        "session_a",
        "session_b",
        "session_a_name",
        "session_b_name",
        "source_roi",
        "target_roi",
        "classification",
        "predicted_count",
        "reference_count",
        "true_positive_count",
        "false_positive_count",
        "false_negative_count",
        "policy_threshold_method",
        "policy_iou_distance_threshold",
        "policy_transform_type",
        "policy_cell_probability_threshold",
        "subject_pairwise_f1",
        "subject_complete_track_f1",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + extras


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
