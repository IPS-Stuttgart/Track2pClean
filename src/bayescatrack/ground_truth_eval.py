"""Ground-truth evaluation helpers for Track2p-style track tables.

This module uses BayesCaTrack matching helpers to load Track2p-style ground-truth
and prediction CSV files and score predicted track tables with exact full-track
and prefix-horizon metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import operator
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .matching import build_track_rows_from_matches

_MISSING_VALUE_STRINGS = {"", "na", "nan", "none", "null", "-"}
_TRACK_ID_HEADERS = {
    "track_id",
    "track",
    "id",
    "gt_id",
    "gt_track",
    "trackid",
}
_LONG_TRACK_HEADERS = _TRACK_ID_HEADERS | {"trajectory", "traj", "trajectory_id"}
_LONG_SESSION_HEADERS = {"session", "session_name", "day", "dataset", "recording"}
_LONG_ROI_HEADERS = {
    "roi",
    "roi_index",
    "roi_idx",
    "cell",
    "cell_id",
    "cell_index",
    "s2p_index",
    "s2p_idx",
    "index",
}
_SESSION_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}(?:_.+)?$")


@dataclass(frozen=True)
class TrackTable:
    """Track table with one row per track and one column per session."""

    session_names: tuple[str, ...]
    tracks: np.ndarray

    def __post_init__(self) -> None:
        session_names = _normalize_session_names(
            self.session_names,
            name="session_names",
        )
        tracks = np.asarray(self.tracks, dtype=int)
        if tracks.ndim != 2:
            raise ValueError("tracks must have shape (n_tracks, n_sessions)")
        if tracks.shape[1] != len(session_names):
            raise ValueError(
                "tracks second dimension must equal the number of session names"
            )
        object.__setattr__(self, "session_names", session_names)
        object.__setattr__(self, "tracks", tracks)

    @property
    def n_tracks(self) -> int:
        return int(self.tracks.shape[0])

    @property
    def n_sessions(self) -> int:
        return int(self.tracks.shape[1])

    def aligned_to(self, session_names: Sequence[str]) -> "TrackTable":
        """Return a copy with columns reordered to ``session_names``."""
        session_names = _normalize_session_names(
            session_names,
            name="target session_names",
        )
        if set(session_names) != set(self.session_names):
            raise ValueError("session names must match exactly for alignment")
        if session_names == self.session_names:
            return self
        indices = [self.session_names.index(name) for name in session_names]
        return TrackTable(session_names=session_names, tracks=self.tracks[:, indices])

    def row_tuples(
        self,
        *,
        horizon: int | None = None,
        require_complete: bool = False,
    ) -> list[tuple[int, ...]]:
        """Return rows as integer tuples, optionally truncated to ``horizon``."""
        if horizon is None:
            rows = self.tracks
        else:
            if not 1 <= horizon <= self.n_sessions:
                raise ValueError("horizon must be between 1 and the number of sessions")
            rows = self.tracks[:, :horizon]

        tuples: list[tuple[int, ...]] = []
        for row in rows.tolist():
            row_tuple = tuple(int(value) for value in row)
            if require_complete and any(value < 0 for value in row_tuple):
                continue
            tuples.append(row_tuple)
        return tuples


@dataclass(frozen=True)
class TrackEvaluation:
    """Summary of Track2p-style benchmark metrics."""

    complete_tracks: float
    proportion_correct_by_horizon: dict[int, float]
    n_ground_truth_tracks: int
    n_predicted_tracks: int
    n_exact_full_track_matches: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "complete_tracks": float(self.complete_tracks),
            "proportion_correct_by_horizon": {
                str(horizon): float(value)
                for horizon, value in self.proportion_correct_by_horizon.items()
            },
            "n_ground_truth_tracks": int(self.n_ground_truth_tracks),
            "n_predicted_tracks": int(self.n_predicted_tracks),
            "n_exact_full_track_matches": int(self.n_exact_full_track_matches),
        }


def _normalize_session_names(session_names: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(session_names, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of session names, not a bare string")
    try:
        normalized = tuple(str(session_name) for session_name in session_names)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of session names") from exc
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    duplicate_names = [
        session_name
        for session_name, count in Counter(normalized).items()
        if count > 1
    ]
    if duplicate_names:
        duplicate_text = ", ".join(repr(session_name) for session_name in duplicate_names)
        raise ValueError(
            f"{name} must contain unique session names; duplicate values: "
            f"{duplicate_text}"
        )
    return normalized


def _normalize_header(header: str) -> str:
    return header.strip().lower().replace(" ", "_")


def _is_missing_text(value: object) -> bool:
    return _normalize_header(str(value).strip()) in _MISSING_VALUE_STRINGS


def _parse_required_long_text(value: object, *, column_name: str) -> str:
    text = str(value).strip()
    if _is_missing_text(text):
        raise ValueError(f"Long-format CSV contains missing {column_name}")
    return text


def _parse_roi_value(  # pylint: disable=too-many-return-statements
    value: str | int | float | None,
) -> int:
    if value is None:
        return -1
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return -1
        if float(value).is_integer():
            return int(value)
        raise ValueError(f"ROI index must be integer-like, got {value!r}")

    text = str(value).strip()
    if _is_missing_text(text):
        return -1
    number = float(text)
    if np.isnan(number):
        return -1
    if not float(number).is_integer():
        raise ValueError(f"ROI index must be integer-like, got {value!r}")
    return int(number)


def _parse_semicolon_roi_values(value: str, *, n_sessions: int) -> list[int]:
    """Parse Track2p ground-truth rows stored as ``roi;roi;...`` strings."""

    parts = str(value).strip().split(";")
    if len(parts) < n_sessions:
        parts.extend([""] * (n_sessions - len(parts)))
    extra_parts = parts[n_sessions:]
    nonempty_parts = [part for part in parts if not _is_missing_text(part)]
    nonempty_extra_parts = [part for part in extra_parts if not _is_missing_text(part)]
    if nonempty_extra_parts:
        raise ValueError(
            "semicolon-encoded track has more non-empty entries than sessions "
            f"({len(nonempty_parts)} non-empty values, {n_sessions} sessions): {value!r}"
        )
    return [_parse_roi_value(part) for part in parts[:n_sessions]]


def _infer_semicolon_session_count(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> int | None:
    """Infer the number of sessions for ``track_id,track`` semicolon CSVs."""

    widths: list[int] = []
    for row in rows:
        data_headers: list[str] = []
        for header in headers:
            value = str(row.get(header, "")).strip()
            if _normalize_header(header) in _TRACK_ID_HEADERS and ";" not in value:
                continue
            data_headers.append(header)
        values = [str(row.get(header, "")).strip() for header in data_headers]
        semicolon_values = [value for value in values if ";" in value]
        if not semicolon_values:
            continue
        nonempty_plain_values = [
            value
            for value in values
            if ";" not in value and not _is_missing_text(value)
        ]
        if len(semicolon_values) == 1 and not nonempty_plain_values:
            widths.append(len(semicolon_values[0].split(";")))
            continue
        raise ValueError(
            "CSV row mixes semicolon-encoded tracks with per-session ROI values"
        )

    if not widths:
        return None
    return max(widths)


def _semicolon_encoded_row(
    headers: Sequence[str],
    row: Mapping[str, str],
    *,
    n_sessions: int,
) -> list[int] | None:
    """Return a semicolon-encoded track row, if the CSV row uses that representation."""

    data_headers: list[str] = []
    for header in headers:
        value = str(row.get(header, "")).strip()
        if _normalize_header(header) in _TRACK_ID_HEADERS and ";" not in value:
            continue
        data_headers.append(header)
    values = [str(row.get(header, "")).strip() for header in data_headers]
    semicolon_values = [value for value in values if ";" in value]
    if not semicolon_values:
        return None
    nonempty_plain_values = [
        value for value in values if ";" not in value and not _is_missing_text(value)
    ]
    if len(semicolon_values) == 1 and not nonempty_plain_values:
        return _parse_semicolon_roi_values(semicolon_values[0], n_sessions=n_sessions)
    raise ValueError(
        "CSV row mixes semicolon-encoded tracks with per-session ROI values"
    )


def _rows_from_csv(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")
        fieldnames = [str(name) for name in reader.fieldnames]
        rows = [{str(key): value for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"CSV file {csv_path} contains no data rows")
    return fieldnames, rows


def _looks_like_long_format(headers: Sequence[str]) -> bool:
    normalized = {_normalize_header(header) for header in headers}
    return (
        bool(normalized & _LONG_TRACK_HEADERS)
        and bool(normalized & _LONG_SESSION_HEADERS)
        and bool(normalized & _LONG_ROI_HEADERS)
    )


def _find_matching_header(headers: Sequence[str], candidates: set[str]) -> str | None:
    for header in headers:
        if _normalize_header(header) in candidates:
            return header
    return None


def _load_long_format(  # pylint: disable=too-many-locals
    headers: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    session_names: Sequence[str] | None,
) -> TrackTable:
    track_header = _find_matching_header(headers, _LONG_TRACK_HEADERS)
    session_header = _find_matching_header(headers, _LONG_SESSION_HEADERS)
    roi_header = _find_matching_header(headers, _LONG_ROI_HEADERS)
    if track_header is None or session_header is None or roi_header is None:
        raise ValueError("could not infer track/session/roi columns from long CSV")

    if session_names is None:
        ordered_sessions: list[str] = []
        seen_sessions: set[str] = set()
        for row in rows:
            session_name = _parse_required_long_text(
                row[session_header], column_name="session name"
            )
            if session_name not in seen_sessions:
                seen_sessions.add(session_name)
                ordered_sessions.append(session_name)
        session_names = ordered_sessions
    else:
        session_names = _normalize_session_names(session_names, name="session_names")

    session_to_index = {name: index for index, name in enumerate(session_names)}
    grouped: dict[str, np.ndarray] = {}
    for row in rows:
        track_id = _parse_required_long_text(row[track_header], column_name="track id")
        session_name = _parse_required_long_text(
            row[session_header], column_name="session name"
        )
        if session_name not in session_to_index:
            continue
        roi_index = _parse_roi_value(row[roi_header])
        session_index = session_to_index[session_name]
        if track_id not in grouped:
            grouped[track_id] = np.full((len(session_names),), -1, dtype=int)
        current_roi = int(grouped[track_id][session_index])
        if current_roi != -1 and current_roi != roi_index:
            raise ValueError(
                "Long-format CSV contains conflicting ROI entries for "
                f"track {track_id!r} in session {session_name!r}: "
                f"{current_roi} and {roi_index}"
            )
        grouped[track_id][session_index] = roi_index

    ordered_track_ids = sorted(grouped)
    if ordered_track_ids:
        tracks = np.vstack([grouped[track_id] for track_id in ordered_track_ids])
    else:
        tracks = np.zeros((0, len(session_names)), dtype=int)
    return TrackTable(session_names=tuple(session_names), tracks=tracks)


def _load_wide_format(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    session_names: Sequence[str] | None,
) -> TrackTable:
    if session_names is None:
        semicolon_session_count = _infer_semicolon_session_count(headers, rows)
        if semicolon_session_count is not None:
            session_names = [
                f"session_{index}" for index in range(semicolon_session_count)
            ]
        else:
            candidate_headers: list[str] = []
            for header in headers:
                if _normalize_header(header) in _TRACK_ID_HEADERS:
                    continue
                candidate_headers.append(header)
            if not candidate_headers:
                raise ValueError("could not infer any session columns from wide CSV")
            session_names = [str(name) for name in candidate_headers]
    else:
        session_names = _normalize_session_names(session_names, name="session_names")

    tracks = np.full((len(rows), len(session_names)), -1, dtype=int)
    for row_index, row in enumerate(rows):
        semicolon_values = _semicolon_encoded_row(
            headers, row, n_sessions=len(session_names)
        )
        if semicolon_values is not None:
            tracks[row_index, :] = semicolon_values
            continue
        for session_index, session_name in enumerate(session_names):
            if session_name not in row:
                raise ValueError(
                    f"session column {session_name!r} not present in CSV row"
                )
            tracks[row_index, session_index] = _parse_roi_value(row[session_name])
    return TrackTable(session_names=tuple(session_names), tracks=tracks)


def _looks_like_track2p_session_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "suite2p").exists() or (path / "data_npy").exists():
        return True
    return _SESSION_DIR_PATTERN.match(path.name) is not None


def _infer_subject_session_names_from_ground_truth_path(
    csv_path: str | Path,
) -> tuple[str, ...]:
    """Infer sibling Track2p session folders for a subject-level ground_truth.csv."""

    parent = Path(csv_path).parent
    if not parent.exists() or not parent.is_dir():
        return ()
    session_dirs = [
        child
        for child in parent.iterdir()
        if child.name != "track2p" and _looks_like_track2p_session_dir(child)
    ]
    session_dirs.sort(key=lambda path: path.name)
    return tuple(path.name for path in session_dirs)


def load_track_table_csv(
    csv_path: str | Path,
    *,
    session_names: Sequence[str] | None = None,
) -> TrackTable:
    """Load a track table from a wide or long CSV file."""
    headers, rows = _rows_from_csv(csv_path)
    if _looks_like_long_format(headers):
        return _load_long_format(headers, rows, session_names)
    return _load_wide_format(headers, rows, session_names)


def load_track2p_ground_truth_csv(
    csv_path: str | Path,
    *,
    session_names: Sequence[str] | None = None,
) -> TrackTable:
    """Load a Track2p ``ground_truth.csv`` file.

    If ``session_names`` is omitted and the CSV sits in a Track2p subject folder,
    sibling session directory names are used. This prevents semicolon-encoded
    ``track_id,track`` files from being mistaken for one-session wide CSVs.
    """

    if session_names is None:
        inferred_session_names = _infer_subject_session_names_from_ground_truth_path(
            csv_path
        )
        if inferred_session_names:
            session_names = inferred_session_names
    return load_track_table_csv(csv_path, session_names=session_names)


def tracks_from_consecutive_matches(
    session_names: Sequence[str],
    matches: Sequence[
        Mapping[int, int]
        | Sequence[tuple[int, int]]
        | np.ndarray
        | tuple[Sequence[int], Sequence[int]]
    ],
    *,
    start_roi_indices: Sequence[int] | None = None,
) -> TrackTable:
    """Reconstruct wide tracks from consecutive pairwise assignments.

    When ``start_roi_indices`` is omitted, only ROIs that appear in the first
    consecutive match are used as seed rows. Pass explicit seed ROI indices when
    unmatched start-session ROIs must be retained in the output table.
    """

    session_names = _normalize_session_names(session_names, name="session_names")
    if start_roi_indices is None:
        start_roi_indices = _infer_start_roi_indices_from_first_match(matches)
    track_rows = build_track_rows_from_matches(
        session_names,
        matches,
        start_roi_indices=start_roi_indices,
        fill_value=-1,
    )
    return TrackTable(session_names=session_names, tracks=track_rows)


def _infer_start_roi_indices_from_first_match(
    matches: Sequence[
        Mapping[int, int]
        | Sequence[tuple[int, int]]
        | np.ndarray
        | tuple[Sequence[int], Sequence[int]]
    ],
) -> list[int]:
    if not matches:
        return []

    first_match = matches[0]
    if isinstance(first_match, Mapping):
        start_values = list(first_match.keys())
    elif isinstance(first_match, tuple) and len(first_match) == 2:
        start_values = list(first_match[0])
    else:
        match_array = np.asarray(first_match, dtype=object)
        if match_array.size == 0:
            return []
        if match_array.ndim != 2 or match_array.shape[1] != 2:
            raise TypeError("unsupported match representation")
        start_values = match_array[:, 0].tolist()

    return sorted({_parse_start_roi_index(value) for value in start_values})


def _parse_start_roi_index(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("start ROI indices must be non-negative integers")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError("start ROI indices must be non-negative integers")
        normalized = int(value)
    else:
        try:
            normalized = operator.index(value)
        except TypeError as exc:
            raise ValueError("start ROI indices must be non-negative integers") from exc

    normalized = int(normalized)
    if normalized < 0:
        raise ValueError("start ROI indices must be non-negative integers")
    return normalized


def _align_prediction_to_ground_truth(
    ground_truth: TrackTable,
    prediction: TrackTable,
) -> TrackTable:
    if set(ground_truth.session_names) != set(prediction.session_names):
        raise ValueError(
            "ground truth and prediction must refer to the same set of sessions"
        )
    return prediction.aligned_to(ground_truth.session_names)


def _row_counter(
    track_table: TrackTable,
    *,
    horizon: int | None = None,
    require_complete: bool = False,
) -> Counter[tuple[int, ...]]:
    return Counter(
        track_table.row_tuples(
            horizon=horizon,
            require_complete=require_complete,
        )
    )


def _multiset_intersection_size(
    left: Counter[tuple[int, ...]],
    right: Counter[tuple[int, ...]],
) -> int:
    return int(sum((left & right).values()))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)


def complete_tracks_score(ground_truth: TrackTable, prediction: TrackTable) -> float:
    """Return the Track2p 'complete tracks' (CT) score.

    This is the F1 score where positives are exact full-track reconstructions.
    Duplicate predicted tracks therefore still count as false positives.
    """
    prediction = _align_prediction_to_ground_truth(ground_truth, prediction)
    ground_truth_rows = _row_counter(ground_truth, require_complete=True)
    prediction_rows = _row_counter(prediction, require_complete=True)
    true_positives = _multiset_intersection_size(ground_truth_rows, prediction_rows)
    predicted_total = int(sum(prediction_rows.values()))
    ground_truth_total = int(sum(ground_truth_rows.values()))
    false_positives = predicted_total - true_positives
    false_negatives = ground_truth_total - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return (
            1.0
            if ground_truth.tracks.shape[0] == 0 and prediction.tracks.shape[0] == 0
            else 0.0
        )
    return _safe_ratio(2.0 * true_positives, denominator)


def proportion_correct_by_horizon(
    ground_truth: TrackTable,
    prediction: TrackTable,
) -> dict[int, float]:
    """Return the fraction of complete GT prefixes reconstructed at each horizon."""
    prediction = _align_prediction_to_ground_truth(ground_truth, prediction)
    result: dict[int, float] = {}
    for horizon in range(2, ground_truth.n_sessions + 1):
        ground_truth_rows = _row_counter(
            ground_truth,
            horizon=horizon,
            require_complete=True,
        )
        prediction_rows = _row_counter(
            prediction,
            horizon=horizon,
            require_complete=True,
        )
        denominator = int(sum(ground_truth_rows.values()))
        if denominator == 0:
            result[horizon] = (
                1.0
                if ground_truth.tracks.shape[0] == 0 and prediction.tracks.shape[0] == 0
                else 0.0
            )
            continue
        correctly_reconstructed = _multiset_intersection_size(
            ground_truth_rows,
            prediction_rows,
        )
        result[horizon] = _safe_ratio(correctly_reconstructed, denominator)
    return result


def evaluate_track_table_prediction(
    ground_truth: TrackTable,
    prediction: TrackTable,
) -> TrackEvaluation:
    """Compute the Track2p benchmark metrics for one prediction."""
    prediction = _align_prediction_to_ground_truth(ground_truth, prediction)
    exact_matches = _multiset_intersection_size(
        _row_counter(ground_truth, require_complete=True),
        _row_counter(prediction, require_complete=True),
    )
    return TrackEvaluation(
        complete_tracks=complete_tracks_score(ground_truth, prediction),
        proportion_correct_by_horizon=proportion_correct_by_horizon(
            ground_truth,
            prediction,
        ),
        n_ground_truth_tracks=ground_truth.n_tracks,
        n_predicted_tracks=prediction.n_tracks,
        n_exact_full_track_matches=exact_matches,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted Track2p-style tracks against ground truth."
    )
    parser.add_argument("ground_truth_csv", help="Path to Track2p ground_truth.csv")
    parser.add_argument("prediction_csv", help="Path to predicted tracks CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ground_truth = load_track2p_ground_truth_csv(args.ground_truth_csv)
    prediction = load_track_table_csv(
        args.prediction_csv,
        session_names=ground_truth.session_names,
    )
    evaluation = evaluate_track_table_prediction(ground_truth, prediction)
    print(json.dumps(evaluation.to_json_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
