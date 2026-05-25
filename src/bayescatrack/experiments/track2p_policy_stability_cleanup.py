"""Stability-based prune-only cleanup for Track2p-policy tracks.

The promoted Track2p-policy row is already close to Track2p but still creates
some false-positive continuations. This runner keeps the same policy links as
the base prediction and removes only bridges that are not stable under nearby
IoU-distance thresholds. No rescue links are added.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.evaluation.track2p_metrics import normalize_track_matrix
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_emulation_benchmark import (
    ThresholdMethod,
    emulate_track2p_tracks,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)

TRACK2P_POLICY_STABILITY_CLEANUP_METHOD = "track2p-policy-stability-cleanup"
Edge = tuple[int, int, int, int]


@dataclass(frozen=True)
class StabilityCleanupConfig:
    """Configuration for threshold-stability pruning."""

    iou_distance_thresholds: tuple[float, ...] = (10.0, 12.0, 14.0)
    base_iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD
    min_support_fraction: float = 2.0 / 3.0
    min_support_votes: int | None = None
    min_side_observations: int = 2

    def __post_init__(self) -> None:
        thresholds = tuple(float(value) for value in self.iou_distance_thresholds)
        if not thresholds:
            raise ValueError("iou_distance_thresholds must not be empty")
        for value in thresholds:
            _require_finite_nonnegative(value, name="iou_distance_thresholds")
        _require_finite_nonnegative(
            self.base_iou_distance_threshold, name="base_iou_distance_threshold"
        )
        if not 0.0 < float(self.min_support_fraction) <= 1.0:
            raise ValueError("min_support_fraction must lie in (0, 1]")
        min_support_votes = (
            None
            if self.min_support_votes is None
            else _positive_int_value(self.min_support_votes, name="min_support_votes")
        )
        min_side_observations = _positive_int_value(
            self.min_side_observations, name="min_side_observations"
        )
        object.__setattr__(self, "iou_distance_thresholds", thresholds)
        object.__setattr__(
            self,
            "base_iou_distance_threshold",
            float(self.base_iou_distance_threshold),
        )
        object.__setattr__(
            self, "min_support_fraction", float(self.min_support_fraction)
        )
        object.__setattr__(self, "min_side_observations", min_side_observations)
        object.__setattr__(self, "min_support_votes", min_support_votes)

    @property
    def ensemble_iou_distance_thresholds(self) -> tuple[float, ...]:
        """Return thresholds used for voting, always including the base."""

        values: list[float] = [float(self.base_iou_distance_threshold)]
        for value in self.iou_distance_thresholds:
            if not any(math.isclose(value, existing) for existing in values):
                values.append(float(value))
        return tuple(values)

    @property
    def required_support_votes(self) -> int:
        """Return the number of ensemble predictions needed to keep a bridge."""

        n_thresholds = len(self.ensemble_iou_distance_thresholds)
        if self.min_support_votes is not None:
            return min(int(self.min_support_votes), n_thresholds)
        return max(1, math.ceil(float(self.min_support_fraction) * n_thresholds))


def run_track2p_policy_stability_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: StabilityCleanupConfig | None = None,
) -> list[SubjectBenchmarkResult]:
    """Run stability-based prune-only cleanup over all Track2p subjects."""

    cleanup = cleanup_config or StabilityCleanupConfig()
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

    results: list[SubjectBenchmarkResult] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        base_prediction = emulate_track2p_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=cleanup.base_iou_distance_threshold,
        )
        ensemble_predictions = tuple(
            emulate_track2p_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=threshold,
            )
            for threshold in cleanup.ensemble_iou_distance_thresholds
        )
        cleaned, split_rows = apply_stability_splits_to_tracks(
            base_prediction,
            edge_support_counts(ensemble_predictions),
            required_support_votes=cleanup.required_support_votes,
            min_side_observations=cleanup.min_side_observations,
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        scores = {
            **scores,
            "track2p_policy_variant": TRACK2P_POLICY_STABILITY_CLEANUP_METHOD,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_base_iou_distance_threshold": float(
                cleanup.base_iou_distance_threshold
            ),
            "track2p_policy_stability_iou_distance_thresholds": json.dumps(
                list(cleanup.ensemble_iou_distance_thresholds)
            ),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_policy_stability_min_support_fraction": float(
                cleanup.min_support_fraction
            ),
            "track2p_policy_stability_min_support_votes": int(
                cleanup.required_support_votes
            ),
            "track2p_policy_stability_min_side_observations": int(
                cleanup.min_side_observations
            ),
            "track2p_policy_stability_applied_splits": int(len(split_rows)),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant="Track2p-policy stability cleanup",
                method=cast(Any, TRACK2P_POLICY_STABILITY_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
    return results


def edge_support_counts(track_matrices: Sequence[Any]) -> Counter[Edge]:
    """Count in how many prediction matrices each adjacent edge appears."""

    counts: Counter[Edge] = Counter()
    for matrix_values in track_matrices:
        seen_this_prediction: set[Edge] = set()
        for row in _normalize_int_track_matrix(matrix_values):
            seen_this_prediction.update(_track_row_edges(row))
        counts.update(seen_this_prediction)
    return counts


def apply_stability_splits_to_tracks(
    predicted_track_matrix: Any,
    support_counts: Mapping[Edge, int],
    *,
    required_support_votes: int,
    min_side_observations: int = 2,
) -> tuple[np.ndarray, tuple[dict[str, int], ...]]:
    """Split base tracks at adjacent bridges with insufficient support."""

    required_support_votes = _positive_int_value(
        required_support_votes, name="required_support_votes"
    )
    min_side_observations = _positive_int_value(
        min_side_observations, name="min_side_observations"
    )

    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    output: list[np.ndarray] = []
    split_rows: list[dict[str, int]] = []
    for track_id, row in enumerate(predicted):
        candidates: list[tuple[int, int, Edge]] = []
        for split_index, edge in _track_row_edges_with_split_indices(row):
            support = int(support_counts.get(edge, 0))
            if support < required_support_votes:
                candidates.append((support, split_index, edge))
        selected_splits = _select_safe_splits(
            row,
            candidates,
            required_support_votes=int(required_support_votes),
            min_side_observations=int(min_side_observations),
        )
        if not selected_splits:
            output.append(np.asarray(row, dtype=int).copy())
            continue
        for support, split_index, edge in candidates:
            if split_index not in selected_splits:
                continue
            split_rows.append(
                {
                    "predicted_track_id": int(track_id),
                    "split_session_a": int(split_index),
                    "split_session_b": int(split_index + 1),
                    "source_roi": int(edge[2]),
                    "target_roi": int(edge[3]),
                    "support_votes": int(support),
                    "required_support_votes": int(required_support_votes),
                }
            )
        output.extend(_split_row_at_indices(row, selected_splits))

    if not output:
        return predicted[:0], tuple(split_rows)
    return np.vstack(output).astype(int, copy=False), tuple(split_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for stability cleanup."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-stability-cleanup",
        description=(
            "Run prune-only Track2p-policy cleanup by splitting bridges that "
            "are unstable across nearby IoU-distance thresholds."
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
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    parser.add_argument(
        "--base-iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--stability-iou-distance-thresholds",
        type=_float_tuple_arg,
        default=(10.0, 12.0, 14.0),
        help="Comma-separated IoU-distance thresholds used for stability voting.",
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument("--min-support-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-support-votes", type=int, default=None)
    parser.add_argument("--min-side-observations", type=int, default=2)
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy stability cleanup CLI."""

    args = build_arg_parser().parse_args(argv)
    cleanup_config = StabilityCleanupConfig(
        iou_distance_thresholds=tuple(args.stability_iou_distance_thresholds),
        base_iou_distance_threshold=args.base_iou_distance_threshold,
        min_support_fraction=args.min_support_fraction,
        min_support_votes=args.min_support_votes,
        min_side_observations=args.min_side_observations,
    )
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=(
            args.allow_track2p_as_reference_for_smoke_test
        ),
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=False,
        weighted_centroids=False,
        exclude_overlapping_pixels=False,
    )
    results = run_track2p_policy_stability_cleanup(
        config,
        threshold_method=cast(Literal["otsu", "min"], args.threshold_method),
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
        cleanup_config=cleanup_config,
    )
    rows = [result.to_dict() for result in results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    return 0


def _normalize_int_track_matrix(track_matrix: Any) -> np.ndarray:
    matrix = normalize_track_matrix(track_matrix)
    output = np.full(matrix.shape, -1, dtype=int)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if _valid_roi(value):
                output[row_index, column_index] = int(value)
    return output


def _track_row_edges(row: np.ndarray) -> set[Edge]:
    return {edge for _, edge in _track_row_edges_with_split_indices(row)}


def _track_row_edges_with_split_indices(
    row: np.ndarray,
) -> tuple[tuple[int, Edge], ...]:
    edges: list[tuple[int, Edge]] = []
    for session_index in range(max(0, row.size - 1)):
        source = row[session_index]
        target = row[session_index + 1]
        if source < 0 or target < 0:
            continue
        edges.append(
            (
                int(session_index),
                (
                    int(session_index),
                    int(session_index + 1),
                    int(source),
                    int(target),
                ),
            )
        )
    return tuple(edges)


def _select_safe_splits(
    row: np.ndarray,
    candidates: Sequence[tuple[int, int, Edge]],
    *,
    required_support_votes: int,
    min_side_observations: int,
) -> tuple[int, ...]:
    """Return the highest-deficit compatible split set.

    A greedy pass can keep an unstable bridge because an earlier selected split
    creates a short fragment. This optimizer scores each candidate by missing
    support votes and selects the maximum-scoring split set whose fragments all
    retain the requested number of observations.
    """

    if not candidates:
        return ()
    support_by_split: dict[int, int] = {}
    for support, split_index, _ in candidates:
        index = int(split_index)
        support_by_split[index] = min(
            int(support),
            support_by_split.get(index, int(support)),
        )
    split_indices = tuple(sorted(support_by_split))
    split_weights = {
        split_index: max(1, int(required_support_votes) - support)
        for split_index, support in support_by_split.items()
    }
    cache: dict[tuple[int, int], tuple[tuple[int, int, int], tuple[int, ...]]] = {}
    impossible = (-(10**9), -(10**9), -(10**9))

    def best_from(
        fragment_start: int,
        candidate_start: int,
    ) -> tuple[tuple[int, int, int], tuple[int, ...]]:
        key = (int(fragment_start), int(candidate_start))
        if key in cache:
            return cache[key]

        best_score = impossible
        best_splits: tuple[int, ...] = ()
        if _observation_count(row, fragment_start, row.size - 1) >= int(
            min_side_observations
        ):
            best_score = (0, 0, 0)

        for candidate_pos in range(candidate_start, len(split_indices)):
            split_index = split_indices[candidate_pos]
            if split_index < fragment_start:
                continue
            if _observation_count(row, fragment_start, split_index) < int(
                min_side_observations
            ):
                continue
            tail_score, tail_splits = best_from(split_index + 1, candidate_pos + 1)
            if tail_score == impossible:
                continue
            score = (
                tail_score[0] + split_weights[split_index],
                tail_score[1] + 1,
                tail_score[2] - split_index,
            )
            if score > best_score:
                best_score = score
                best_splits = (split_index, *tail_splits)

        cache[key] = (best_score, best_splits)
        return cache[key]

    _, selected = best_from(0, 0)
    return selected


def _fragments_have_min_observations(
    row: np.ndarray, split_indices: set[int], *, min_observations: int
) -> bool:
    return all(
        count >= int(min_observations)
        for count in _fragment_observation_counts(row, split_indices)
    )


def _fragment_observation_counts(
    row: np.ndarray, split_indices: set[int]
) -> tuple[int, ...]:
    counts: list[int] = []
    start = 0
    for split_index in sorted(split_indices):
        counts.append(int(np.sum(row[start : split_index + 1] >= 0)))
        start = int(split_index) + 1
    counts.append(int(np.sum(row[start:] >= 0)))
    return tuple(counts)


def _observation_count(row: np.ndarray, start: int, stop: int) -> int:
    if int(stop) < int(start):
        return 0
    return int(np.sum(row[int(start) : int(stop) + 1] >= 0))


def _split_row_at_indices(
    row: np.ndarray, split_indices: Sequence[int]
) -> tuple[np.ndarray, ...]:
    if not split_indices:
        return (np.asarray(row, dtype=int).copy(),)
    fragments: list[np.ndarray] = []
    start = 0
    for split_index in sorted(set(int(index) for index in split_indices)):
        fragment = np.full(row.shape, -1, dtype=int)
        fragment[start : split_index + 1] = row[start : split_index + 1]
        if np.any(fragment >= 0):
            fragments.append(fragment)
        start = split_index + 1
    fragment = np.full(row.shape, -1, dtype=int)
    fragment[start:] = row[start:]
    if np.any(fragment >= 0):
        fragments.append(fragment)
    return tuple(fragments)


def _valid_roi(value: Any) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric >= 0.0)


def _require_finite_nonnegative(value: float, *, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _positive_int_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, int):
        numeric = value
    elif isinstance(value, float) and value.is_integer():
        numeric = int(value)
    else:
        raise ValueError(f"{name} must be a positive integer")
    if numeric <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _float_tuple_arg(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        tokens = tuple(token.strip() for token in value.split(",") if token.strip())
        if not tokens:
            raise argparse.ArgumentTypeError("expected at least one float")
        try:
            return tuple(float(token) for token in tokens)
        except ValueError as exc:  # pragma: no cover - argparse reports this path
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return tuple(float(item) for item in value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
