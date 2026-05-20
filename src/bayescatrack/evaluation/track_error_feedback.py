"""Convert track-matrix errors into feedback rows for calibration loops."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .complete_track_scores import normalize_track_matrix

FeedbackStatus = Literal[
    "true_positive",
    "false_positive",
    "false_negative",
    "duplicate_prediction",
]


def link_feedback_rows(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return aggregate false-positive/false-negative link feedback rows."""

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError("Predicted and reference matrices must have equal columns")
    pairs = _session_pairs(predicted.shape[1], session_pairs)
    predicted_counter = _link_counter(predicted, pairs)
    reference_counter = _link_counter(reference, pairs)
    base = dict(metadata or {})
    rows: list[dict[str, Any]] = []
    for link in sorted(set(predicted_counter) | set(reference_counter)):
        pred_count = int(predicted_counter.get(link, 0))
        ref_count = int(reference_counter.get(link, 0))
        tp = min(pred_count, ref_count)
        fp = max(pred_count - ref_count, 0)
        fn = max(ref_count - pred_count, 0)
        duplicate = max(pred_count - 1, 0)
        if tp:
            rows.append(_feedback_row(base, link, "true_positive", tp, pred_count, ref_count))
        if fp:
            rows.append(_feedback_row(base, link, "false_positive", fp, pred_count, ref_count))
        if fn:
            rows.append(_feedback_row(base, link, "false_negative", fn, pred_count, ref_count))
        if duplicate:
            rows.append(
                _feedback_row(
                    base,
                    link,
                    "duplicate_prediction",
                    duplicate,
                    pred_count,
                    ref_count,
                )
            )
    return rows


def feedback_sample_weight(
    status: str,
    *,
    true_positive_weight: float = 1.0,
    false_positive_weight: float = 2.0,
    false_negative_weight: float = 2.0,
    duplicate_prediction_weight: float = 3.0,
) -> float:
    """Return a default calibration weight for one feedback status."""

    weights = {
        "true_positive": true_positive_weight,
        "false_positive": false_positive_weight,
        "false_negative": false_negative_weight,
        "duplicate_prediction": duplicate_prediction_weight,
    }
    if status not in weights:
        raise ValueError(f"Unsupported feedback status: {status!r}")
    weight = float(weights[status])
    if weight < 0.0 or not np.isfinite(weight):
        raise ValueError("feedback weights must be finite and non-negative")
    return weight


def feedback_rows_to_edge_label_overrides(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, int, int], tuple[int, float]]:
    """Return edge label/weight overrides from feedback rows.

    False positives and duplicate predictions become weighted negatives, while
    false negatives and true positives become positives.  This is intended for
    fold-internal refinement, not for held-out evaluation.
    """

    overrides: dict[tuple[int, int, int, int], tuple[int, float]] = {}
    for row in rows:
        edge = (
            int(row["session_a"]),
            int(row["session_b"]),
            int(row["roi_a"]),
            int(row["roi_b"]),
        )
        status = str(row["feedback_status"])
        label = 0 if status in {"false_positive", "duplicate_prediction"} else 1
        weight = feedback_sample_weight(status) * float(row.get("count", 1))
        previous = overrides.get(edge)
        if previous is None or weight > previous[1]:
            overrides[edge] = (label, weight)
    return overrides


def _feedback_row(
    base: Mapping[str, Any],
    link: tuple[int, int, int, int],
    status: FeedbackStatus,
    count: int,
    predicted_count: int,
    reference_count: int,
) -> dict[str, Any]:
    session_a, session_b, roi_a, roi_b = link
    return {
        **dict(base),
        "session_a": int(session_a),
        "session_b": int(session_b),
        "session_gap": int(session_b - session_a),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "feedback_status": status,
        "count": int(count),
        "predicted_count": int(predicted_count),
        "reference_count": int(reference_count),
    }


def _link_counter(
    track_matrix: np.ndarray, session_pairs: Sequence[tuple[int, int]]
) -> Counter[tuple[int, int, int, int]]:
    counter: Counter[tuple[int, int, int, int]] = Counter()
    for session_a, session_b in session_pairs:
        for row in track_matrix:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if roi_a is not None and roi_b is not None:
                counter[(session_a, session_b, int(roi_a), int(roi_b))] += 1
    return counter


def _session_pairs(
    n_sessions: int, pairs: Iterable[tuple[int, int]] | None
) -> tuple[tuple[int, int], ...]:
    if pairs is None:
        return tuple((i, i + 1) for i in range(max(0, n_sessions - 1)))
    normalized = tuple((int(a), int(b)) for a, b in pairs)
    for a, b in normalized:
        if a < 0 or b >= n_sessions or a >= b:
            raise ValueError("session_pairs must be forward pairs within matrix width")
    return normalized


__all__ = (
    "FeedbackStatus",
    "feedback_rows_to_edge_label_overrides",
    "feedback_sample_weight",
    "link_feedback_rows",
)
