"""False-positive/false-negative audit for the promoted Track2p-policy row.

The benchmark results show that the minimum-threshold Track2p policy is the
right default direction, while DP rescue variants add false positives and lose
policy-supported ground-truth edges. This module therefore focuses on a
prune-only diagnostic: it reruns the policy prediction and emits an edge ledger
that identifies exactly which policy edges are true positives, false positives,
or false negatives against manual ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from bayescatrack.evaluation.track2p_metrics import normalize_track_matrix
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    Track2pBenchmarkConfig,
    _filter_tracks_by_seed_rois,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _reference_seed_roi_set,
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
    emulate_track2p_tracks,
    track2p_policy_config,
)

EdgeStatus = Literal["true_positive", "false_positive", "false_negative"]
TrackEdge = tuple[int, int, int, int]


@dataclass(frozen=True)
class PolicyAuditResult:
    """Edge-level audit rows plus compact per-subject summary rows."""

    edge_rows: tuple[dict[str, int | str], ...]
    summary_rows: tuple[dict[str, float | int | str], ...]


def run_track2p_policy_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    cell_probability_threshold: float | None = None,
    transform_type: str | None = None,
) -> PolicyAuditResult:
    """Run Track2p-policy min and return an edge ledger against manual GT."""

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

    edge_rows: list[dict[str, int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy audit requires independent manual GT references"
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
        predicted, reference_matrix = _apply_seed_roi_filter(
            predicted,
            reference_matrix,
            config=policy_config,
        )
        session_names = tuple(session.session_name for session in sessions)
        subject_rows = policy_edge_ledger_rows(
            predicted,
            reference_matrix,
            subject=subject_dir.name,
            session_names=session_names,
            metadata={
                "threshold_method": str(threshold_method),
                "iou_distance_threshold": float(iou_distance_threshold),
                "cell_probability_threshold": float(
                    policy_config.cell_probability_threshold
                ),
                "transform_type": str(policy_config.transform_type),
            },
        )
        edge_rows.extend(subject_rows)
        summary_rows.append(
            _summary_row(
                subject_dir.name,
                subject_rows,
                threshold_method=threshold_method,
                iou_distance_threshold=iou_distance_threshold,
            )
        )
    return PolicyAuditResult(tuple(edge_rows), tuple(summary_rows))


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
            if _valid_roi_index(roi_a) and _valid_roi_index(roi_b):
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
    """Return duplicate-aware TP/FP/FN count rows for predicted-vs-reference edges."""

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


def policy_edge_ledger_rows(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    subject: str,
    session_names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, int | str]]:
    """Return duplicate-aware edge TP/FP/FN ledger rows."""

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError("Predicted and reference matrices must have matching sessions")
    names = _session_names(session_names, n_sessions=predicted.shape[1])
    meta = {
        key: _format_metadata_value(value) for key, value in dict(metadata or {}).items()
    }

    predicted_counter = track_edge_counter(predicted)
    reference_counter = track_edge_counter(reference)
    remaining_reference = Counter(reference_counter)
    rows: list[dict[str, int | str]] = []

    for edge in sorted(predicted_counter):
        count = int(predicted_counter[edge])
        matched = min(count, int(remaining_reference.get(edge, 0)))
        if matched:
            remaining_reference[edge] -= matched
        false_positive = count - matched
        rows.extend(
            _edge_rows(
                edge,
                subject=subject,
                session_names=names,
                status="true_positive",
                count=matched,
                metadata=meta,
            )
        )
        rows.extend(
            _edge_rows(
                edge,
                subject=subject,
                session_names=names,
                status="false_positive",
                count=false_positive,
                metadata=meta,
            )
        )

    for edge in sorted(remaining_reference):
        rows.extend(
            _edge_rows(
                edge,
                subject=subject,
                session_names=names,
                status="false_negative",
                count=int(remaining_reference[edge]),
                metadata=meta,
            )
        )
    return rows


def track_edge_counter(track_matrix: Any) -> Counter[TrackEdge]:
    """Count consecutive-session identity edges in a track matrix."""

    matrix = normalize_track_matrix(track_matrix)
    counter: Counter[TrackEdge] = Counter()
    for session_index in range(max(0, matrix.shape[1] - 1)):
        for row in matrix:
            left = row[session_index]
            right = row[session_index + 1]
            if _valid_roi_index(left) and _valid_roi_index(right):
                counter[
                    (
                        int(session_index),
                        int(session_index + 1),
                        int(cast(Any, left)),
                        int(cast(Any, right)),
                    )
                ] += 1
    return counter


def write_audit_rows(
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
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-audit",
        description="Audit Track2p-policy min false-positive and false-negative edges.",
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
    parser.add_argument("--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        plane_name=args.plane_name,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        include_behavior=False,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    result = run_track2p_policy_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        cell_probability_threshold=args.cell_probability_threshold,
        transform_type=args.transform_type,
    )
    write_audit_rows(result.edge_rows, args.output, output_format=args.format)
    if args.summary_output is not None:
        write_audit_rows(
            result.summary_rows, args.summary_output, output_format=args.format
        )
    return 0


def _apply_seed_roi_filter(
    predicted: Any,
    reference_matrix: np.ndarray,
    *,
    config: Track2pBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_matrix = normalize_track_matrix(predicted)
    if not config.restrict_to_reference_seed_rois:
        return predicted_matrix, reference_matrix
    reference_seed_rois = _reference_seed_roi_set(
        reference_matrix, seed_session=config.seed_session
    )
    return (
        _filter_tracks_by_seed_rois(
            predicted_matrix,
            reference_seed_rois,
            seed_session=config.seed_session,
        ),
        _filter_tracks_by_seed_rois(
            reference_matrix,
            reference_seed_rois,
            seed_session=config.seed_session,
        ),
    )


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


def _edge_rows(
    edge: TrackEdge,
    *,
    subject: str,
    session_names: Sequence[str],
    status: EdgeStatus,
    count: int,
    metadata: Mapping[str, str],
) -> list[dict[str, int | str]]:
    session_a, session_b, roi_a, roi_b = edge
    return [
        {
            "subject": subject,
            "session_a": int(session_a),
            "session_b": int(session_b),
            "session_a_name": str(session_names[session_a]),
            "session_b_name": str(session_names[session_b]),
            "roi_a": int(roi_a),
            "roi_b": int(roi_b),
            "edge_status": status,
            "occurrence": occurrence,
            **dict(metadata),
        }
        for occurrence in range(1, int(count) + 1)
    ]


def _summary_row(
    subject: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_method: str,
    iou_distance_threshold: float,
) -> dict[str, float | int | str]:
    counts = Counter(str(row["edge_status"]) for row in rows)
    tp = int(counts.get("true_positive", 0))
    fp = int(counts.get("false_positive", 0))
    fn = int(counts.get("false_negative", 0))
    denom = 2 * tp + fp + fn
    return {
        "subject": subject,
        "threshold_method": str(threshold_method),
        "iou_distance_threshold": float(iou_distance_threshold),
        "pairwise_true_positives": tp,
        "pairwise_false_positives": fp,
        "pairwise_false_negatives": fn,
        "pairwise_f1": 1.0 if denom == 0 else float(2 * tp / denom),
    }


def _session_names(
    session_names: Sequence[str] | None, *, n_sessions: int
) -> tuple[str, ...]:
    if session_names is None:
        return tuple(str(index) for index in range(n_sessions))
    names = tuple(str(name) for name in session_names)
    if len(names) != n_sessions:
        raise ValueError("session_names must have one entry per track-matrix column")
    return names


def _valid_roi_index(value: object) -> bool:
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


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
