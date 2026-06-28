"""Fixed-precision diagnostics for longitudinal complete-track recovery."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import Any

import numpy as np

from .complete_track_scores import complete_track_set, normalize_track_matrix

_MUTABLE_BYTES_TYPE = getattr(builtins, "byte" "array")

DEFAULT_COMPLETE_TRACK_FIXED_PRECISIONS = (0.9, 0.95, 0.99)
_MISSING_OBSERVATION_STRINGS = frozenset({"", "none", "nan", "null"})
ScoredCompleteTrack = tuple[tuple[int, ...], float]

__all__ = (
    "DEFAULT_COMPLETE_TRACK_FIXED_PRECISIONS",
    "score_complete_tracks_at_fixed_precision",
)


# pylint: disable=too-many-locals
def score_complete_tracks_at_fixed_precision(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    target_precisions: Sequence[float] = DEFAULT_COMPLETE_TRACK_FIXED_PRECISIONS,
    track_scores: Sequence[float] | None = None,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Report complete-track recovery at fixed precision operating points.

    The diagnostic sweeps score thresholds over predicted complete tracks and
    reports the maximum number of true complete tracks retained while satisfying
    each target precision. Higher ``track_scores`` values are treated as more
    reliable. If no scores are supplied, all predicted tracks receive the same
    score, yielding a conservative all-or-nothing operating point.
    """

    predicted_matrix = normalize_track_matrix(predicted_track_matrix)
    reference_matrix = normalize_track_matrix(reference_track_matrix)
    if predicted_matrix.shape[1] != reference_matrix.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )

    selected_sessions = _resolve_session_indices(
        predicted_matrix.shape[1], session_indices
    )
    _resolve_session_indices(reference_matrix.shape[1], selected_sessions)

    targets = _validate_target_precisions(target_precisions)
    if not targets:
        return {}

    scores = _score_array_for_track_matrix(predicted_matrix, track_scores)
    predicted = _scored_complete_tracks(predicted_matrix, scores, selected_sessions)
    reference = complete_track_set(reference_matrix, session_indices=selected_sessions)

    diagnostics: dict[str, float | int] = {}
    for target_precision in targets:
        suffix = _fixed_precision_metric_suffix(target_precision)
        true_positives, predictions, precision, recall, threshold = (
            _best_operating_point(
                predicted,
                reference,
                target_precision=target_precision,
            )
        )
        diagnostics.update(
            {
                f"complete_tracks_at_fixed_precision_{suffix}": true_positives,
                f"complete_track_predictions_at_fixed_precision_{suffix}": predictions,
                f"complete_track_precision_at_fixed_precision_{suffix}": precision,
                f"complete_track_recall_at_fixed_precision_{suffix}": recall,
                f"complete_track_score_threshold_at_fixed_precision_{suffix}": threshold,
            }
        )
    return diagnostics


def _best_operating_point(
    predicted: Sequence[ScoredCompleteTrack],
    reference: set[tuple[int, ...]],
    *,
    target_precision: float,
) -> tuple[int, int, float, float, float]:
    best_rank = (0, 1.0, 0, 0)
    empty_recall = _safe_ratio(0, len(reference))
    best_result = (0, 0, 1.0, empty_recall, float("inf"))
    for threshold in sorted({score for _, score in predicted}, reverse=True):
        retained = [track for track, score in predicted if score >= threshold]
        retained_unique = set(retained)
        true_positives = len(retained_unique.intersection(reference))
        predictions = len(retained)
        false_positives = predictions - true_positives
        false_negatives = len(reference.difference(retained_unique))
        precision = _safe_ratio(true_positives, predictions)
        recall = _safe_ratio(true_positives, true_positives + false_negatives)
        rank = (true_positives, precision, -false_positives, predictions)
        if precision >= target_precision and rank > best_rank:
            best_rank = rank
            best_result = (
                true_positives,
                predictions,
                precision,
                recall,
                float(threshold),
            )
    return best_result


def _scored_complete_tracks(
    matrix: np.ndarray,
    track_scores: np.ndarray,
    selected_sessions: Sequence[int],
) -> list[ScoredCompleteTrack]:
    scored_tracks: list[ScoredCompleteTrack] = []
    for row, score in zip(matrix, track_scores, strict=True):
        track = _complete_track_tuple(row, selected_sessions)
        if track is not None:
            scored_tracks.append((track, float(score)))
    return scored_tracks


def _complete_track_tuple(
    row: np.ndarray,
    selected_sessions: Sequence[int],
) -> tuple[int, ...] | None:
    roi_values: list[int] = []
    for session_idx in selected_sessions:
        roi_index = _roi_index_or_none(row[session_idx])
        if roi_index is None:
            return None
        roi_values.append(roi_index)
    return tuple(roi_values)


def _roi_index_or_none(value: object) -> int | None:
    if _is_missing_observation(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"track matrix contains boolean ROI index: {value!r}; "
            "ROI observations must be integer-like or missing"
        )
    if isinstance(value, (int, np.integer)):
        roi_index = int(value)
        return roi_index if roi_index >= 0 else None
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            return None
        if numeric.is_integer():
            return int(numeric)
        raise ValueError(
            f"track matrix contains non-integer ROI index: {value!r}; "
            "ROI observations must be integer-like or missing"
        )
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _MISSING_OBSERVATION_STRINGS:
            return None
        try:
            roi_index = int(text, 10)
        except ValueError as exc:
            raise ValueError(
                f"track matrix contains non-integer ROI index: {value!r}; "
                "ROI observations must be integer-like or missing"
            ) from exc
        return roi_index if roi_index >= 0 else None
    raise ValueError(
        f"track matrix contains non-integer ROI index: {value!r}; "
        "ROI observations must be integer-like or missing"
    )


def _is_missing_observation(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_OBSERVATION_STRINGS
    return isinstance(value, (float, np.floating)) and np.isnan(value)


def _score_array_for_track_matrix(
    matrix: np.ndarray, track_scores: Sequence[float] | None
) -> np.ndarray:
    if track_scores is None:
        return np.ones((matrix.shape[0],), dtype=float)
    if isinstance(track_scores, (str, bytes, _MUTABLE_BYTES_TYPE)):
        raise ValueError(
            "track_scores must be a sequence of finite real-valued scores, not a bare string-like value"
        )
    raw_scores = np.asarray(track_scores, dtype=object)
    if raw_scores.ndim != 1 or raw_scores.shape[0] != matrix.shape[0]:
        raise ValueError(
            "track_scores must contain exactly one score per predicted track"
        )
    if any(_is_boolean_scalar(score) for score in raw_scores):
        raise ValueError("track_scores must contain finite real-valued scores")
    try:
        scores = np.asarray(raw_scores, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("track_scores must contain finite real-valued scores") from exc
    if scores.ndim != 1 or scores.shape[0] != matrix.shape[0]:
        raise ValueError(
            "track_scores must contain exactly one score per predicted track"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("track_scores must contain only finite values")
    return scores


def _is_boolean_scalar(value: object) -> bool:
    array = np.asarray(value, dtype=object)
    if array.shape != ():
        return False
    return isinstance(array.item(), (bool, np.bool_))


def _resolve_session_indices(
    num_sessions: int, session_indices: Sequence[int] | None
) -> list[int]:
    if session_indices is None:
        return list(range(num_sessions))
    if isinstance(session_indices, (str, bytes, _MUTABLE_BYTES_TYPE)):
        raise ValueError(
            "session_indices must be a sequence of integer-like indices, not a bare string-like value"
        )
    try:
        iterator = iter(session_indices)
    except TypeError as exc:
        raise ValueError(
            "session_indices must be a sequence of integer-like indices"
        ) from exc
    selected: list[int] = []
    seen: set[int] = set()
    for candidate in iterator:
        session_idx = _coerce_session_index(candidate)
        if session_idx < 0 or session_idx >= num_sessions:
            raise IndexError(
                f"session index {session_idx} out of bounds for {num_sessions} sessions"
            )
        if session_idx in seen:
            raise ValueError("session_indices must not contain duplicate entries")
        seen.add(session_idx)
        selected.append(session_idx)
    return selected


def _coerce_session_index(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"session_indices contains boolean session index: {value!r}; "
            "Session indices must be integer-like"
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return int(value)
        raise ValueError(
            f"session_indices contains non-integer session index: {value!r}; "
            "Session indices must be integer-like"
        )
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(
                f"session_indices contains non-integer session index: {value!r}; "
                "Session indices must be integer-like"
            ) from exc
    raise ValueError(
        f"session_indices contains non-integer session index: {value!r}; "
        "Session indices must be integer-like"
    )


def _validate_target_precisions(
    target_precisions: Sequence[float],
) -> tuple[float, ...]:
    if isinstance(target_precisions, (str, bytes, _MUTABLE_BYTES_TYPE)):
        raise ValueError(
            "target_precisions must be a sequence of finite numeric values between 0 and 1, not a bare string-like value"
        )
    try:
        iterator = iter(target_precisions)
    except TypeError as exc:
        raise ValueError(
            "target_precisions must be a sequence of finite numeric values between 0 and 1"
        ) from exc

    targets: list[float] = []
    for target_precision in iterator:
        target = _coerce_target_precision(target_precision)
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError(
                "target precisions must be finite numeric values between 0 and 1"
            )
        targets.append(target)
    return tuple(targets)


def _coerce_target_precision(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "target precisions must be finite numeric values between 0 and 1"
        )
    try:
        array_value = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError):
        scalar_value = value
    else:
        if array_value.shape != ():
            raise ValueError(
                "target precisions must be scalar finite numeric values between 0 and 1"
            )
        scalar_value = array_value.item()
        if isinstance(scalar_value, (bool, np.bool_)):
            raise ValueError(
                "target precisions must be finite numeric values between 0 and 1"
            )
    try:
        return float(scalar_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "target precisions must be finite numeric values between 0 and 1"
        ) from exc


def _fixed_precision_metric_suffix(target_precision: float) -> str:
    text = f"{float(target_precision):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "minus_").replace(".", "_")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)
