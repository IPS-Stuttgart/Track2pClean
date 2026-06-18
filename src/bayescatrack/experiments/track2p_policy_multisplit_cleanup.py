"""Guarded multi-bridge cleanup for Track2p-policy tracks.

The first component-cleanup pass can split at the single weakest bridge of a
complete predicted component. Some false complete tracks, however, are mosaics
of several reliable sub-trajectories connected by multiple weak bridges. A
single cut removes the complete-track false positive, but it can still leave a
long fragment with another weak bridge that hurts pairwise precision.

This module keeps the same conservative, unsupervised edge-risk model used by
``track2p_policy_component_audit`` and extends only the post-processing step:
select a globally optimized set of weak bridges per component subject to a
minimum fragment length constraint, then split all selected bridges at once.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _filter_tracks_by_seed_rois,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _reference_seed_roi_set,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentAuditOutput,
    ComponentCleanupConfig,
    _component_edges,
    _diagnostics_by_suite2p_edge,
    _no_prune_config,
    _normalize_int_track_matrix,
    _valid_seed_roi,
    component_audit_rows,
    write_component_rows,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_MULTISPLIT_CLEANUP_METHOD = "track2p-policy-multisplit-cleanup"
_IMPOSSIBLE_SCORE = (-float("inf"), -1)


@dataclass(frozen=True)
class MultiSplitCleanupConfig:
    """Configuration for guarded ranked weak-bridge splitting."""

    component: ComponentCleanupConfig = ComponentCleanupConfig()
    max_splits_per_component: int = 2

    def __post_init__(self) -> None:
        _require_positive_int(
            self.max_splits_per_component, name="max_splits_per_component"
        )


def run_track2p_policy_multisplit_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: MultiSplitCleanupConfig | None = None,
    apply_splits: bool = True,
) -> ComponentAuditOutput:
    """Run Track2p-policy and apply guarded multi-bridge cleanup before scoring."""

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

    cleanup_config = cleanup_config or MultiSplitCleanupConfig()
    results: list[SubjectBenchmarkResult] = []
    component_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy multisplit cleanup requires independent manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=_no_prune_config(),
        )
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        predicted_full = _normalize_int_track_matrix(prediction.tracks)
        predicted_eval, reference_eval, evaluated_track_ids = (
            _evaluated_prediction_rows(
                predicted_full,
                reference_tracks,
                config=policy_config,
            )
        )
        subject_rows = component_audit_rows(
            predicted_eval,
            reference_eval,
            sessions=sessions,
            diagnostics=prediction.diagnostics,
            subject=subject_dir.name,
            config=cleanup_config.component,
            track_ids=evaluated_track_ids,
            seed_session=policy_config.seed_session,
        )
        split_plan = plan_ranked_bridge_splits(
            predicted_eval,
            sessions=sessions,
            diagnostics=prediction.diagnostics,
            config=cleanup_config,
            track_ids=evaluated_track_ids,
        )
        enriched_rows = _enrich_component_rows_with_split_plan(
            subject_rows,
            split_plan,
            apply_splits=apply_splits,
        )
        cleaned = (
            apply_ranked_bridge_splits(predicted_full, split_plan)
            if apply_splits
            else predicted_full
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        candidate_splits = int(sum(len(splits) for splits in split_plan.values()))
        applied_splits = int(candidate_splits if apply_splits else 0)
        scores = {
            **scores,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_component_apply_splits": int(apply_splits),
            "track2p_component_candidate_splits": candidate_splits,
            "track2p_component_applied_splits": applied_splits,
            "track2p_component_max_splits_per_component": int(
                cleanup_config.max_splits_per_component
            ),
            "track2p_component_split_risk_threshold": float(
                cleanup_config.component.split_risk_threshold
            ),
            "track2p_component_split_penalty": float(
                cleanup_config.component.split_penalty
            ),
            "track2p_component_min_side_observations": int(
                cleanup_config.component.min_side_observations
            ),
            "track2p_component_require_complete_track": int(
                cleanup_config.component.require_complete_track
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=(
                    "Track2p-policy guarded multi-bridge component split"
                    if apply_splits
                    else "Track2p-policy guarded multi-bridge component audit"
                ),
                method=cast(Any, TRACK2P_POLICY_MULTISPLIT_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        component_rows.extend(
            _with_metadata(
                enriched_rows,
                {
                    "threshold_method": str(threshold_method),
                    "iou_distance_threshold": float(iou_distance_threshold),
                    "cell_probability_threshold": float(
                        policy_config.cell_probability_threshold
                    ),
                    "transform_type": str(policy_config.transform_type),
                    "cleanup_method": TRACK2P_POLICY_MULTISPLIT_CLEANUP_METHOD,
                },
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(component_rows))


def plan_ranked_bridge_splits(
    predicted_track_matrix: Any,
    *,
    sessions: Sequence[Track2pSession],
    diagnostics: Sequence[Any],
    config: MultiSplitCleanupConfig | None = None,
    track_ids: Sequence[int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Return selected bridge indices keyed by original component id.

    Candidate bridges are scored by the existing unsupervised edge-risk score.
    Unlike a greedy ranked pass, this planner maximizes the total compatible
    split gain under the minimum-fragment-observation constraint. This avoids
    selecting one slightly riskier central bridge when two side bridges would
    remove more total weak-bridge evidence.
    """

    config = config or MultiSplitCleanupConfig()
    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    ids = (
        tuple(range(predicted.shape[0]))
        if track_ids is None
        else tuple(int(track_id) for track_id in track_ids)
    )
    if len(ids) != predicted.shape[0]:
        raise ValueError("track_ids must have one entry per predicted track")
    diagnostic_by_edge = _diagnostics_by_suite2p_edge(sessions, diagnostics)
    split_plan: dict[int, tuple[int, ...]] = {}
    for row_index, track in enumerate(predicted):
        selected = _ranked_split_indices_for_track(
            track,
            diagnostic_by_edge=diagnostic_by_edge,
            config=config,
        )
        if selected:
            split_plan[int(ids[row_index])] = selected
    return split_plan


def apply_ranked_bridge_splits(
    predicted_track_matrix: Any,
    split_plan: Mapping[int, Sequence[int]],
) -> np.ndarray:
    """Apply a precomputed bridge-split plan to a track matrix."""

    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    output: list[np.ndarray] = []
    for component_id, track in enumerate(predicted):
        split_indices = tuple(int(index) for index in split_plan.get(component_id, ()))
        if not split_indices:
            output.append(np.asarray(track, dtype=int).copy())
            continue
        output.extend(split_track_at_bridges(track, split_indices))
    if not output:
        return predicted[:0]
    return np.vstack(output).astype(int, copy=False)


def split_track_at_bridges(
    track: Any, session_indices: Sequence[int]
) -> tuple[np.ndarray, ...]:
    """Split one track at multiple bridges and return masked fragments."""

    row = np.asarray(_normalize_int_track_matrix([track])[0], dtype=int)
    split_indices = tuple(sorted({int(index) for index in session_indices}))
    if any(index < 0 or index >= row.size - 1 for index in split_indices):
        raise IndexError("all session_indices must identify consecutive bridges")
    boundaries = (-1, *split_indices, row.size - 1)
    fragments: list[np.ndarray] = []
    for left_boundary, right_boundary in zip(boundaries[:-1], boundaries[1:]):
        start = int(left_boundary) + 1
        stop = int(right_boundary) + 1
        fragment = np.full(row.shape, -1, dtype=int)
        fragment[start:stop] = row[start:stop]
        if np.any(fragment >= 0):
            fragments.append(fragment)
    return tuple(fragments)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.track2p_policy_multisplit_cleanup",
        description="Run guarded multi-bridge Track2p-policy component cleanup.",
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
    parser.add_argument(
        "--apply-splits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-complete-track", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--threshold-margin-scale", type=float, default=0.10)
    parser.add_argument("--competition-margin-scale", type=float, default=0.20)
    parser.add_argument("--area-ratio-floor", type=float, default=0.45)
    parser.add_argument("--centroid-distance-scale", type=float, default=4.0)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument("--max-splits-per-component", type=int, default=2)
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
    parser.add_argument("--component-output", type=Path, default=None)
    parser.add_argument("--component-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    component_config = ComponentCleanupConfig(
        threshold_margin_scale=args.threshold_margin_scale,
        competition_margin_scale=args.competition_margin_scale,
        area_ratio_floor=args.area_ratio_floor,
        centroid_distance_scale=args.centroid_distance_scale,
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    cleanup_config = MultiSplitCleanupConfig(
        component=component_config,
        max_splits_per_component=args.max_splits_per_component,
    )
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
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    output = run_track2p_policy_multisplit_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
        cleanup_config=cleanup_config,
        apply_splits=args.apply_splits,
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.component_output is not None:
        write_component_rows(
            output.component_rows,
            args.component_output,
            output_format=cast(Literal["csv", "json"], args.component_format),
        )
    return 0


def _evaluated_prediction_rows(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    config: Track2pBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    predicted = _normalize_int_track_matrix(predicted)
    reference = _normalize_int_track_matrix(reference)
    if not config.restrict_to_reference_seed_rois:
        return predicted, reference, tuple(range(predicted.shape[0]))
    reference_seed_rois = _reference_seed_roi_set(
        reference, seed_session=config.seed_session
    )
    keep_indices = tuple(
        index
        for index, row in enumerate(predicted)
        if _valid_seed_roi(row, reference_seed_rois, seed_session=config.seed_session)
    )
    predicted_eval = predicted[np.asarray(keep_indices, dtype=int)]
    reference_eval = _filter_tracks_by_seed_rois(
        reference,
        reference_seed_rois,
        seed_session=config.seed_session,
    )
    return predicted_eval, reference_eval, keep_indices


def _ranked_split_indices_for_track(
    track: np.ndarray,
    *,
    diagnostic_by_edge: Mapping[tuple[int, int, int], Any],
    config: MultiSplitCleanupConfig,
) -> tuple[int, ...]:
    component_config = config.component
    observed = int(np.sum(track >= 0))
    is_complete = bool(observed == int(track.size))
    if component_config.require_complete_track and not is_complete:
        return ()
    edges = _component_edges(track, diagnostic_by_edge, config=component_config)
    candidate_gains: dict[int, float] = {}
    for edge in edges:
        if edge.risk < component_config.split_risk_threshold:
            continue
        gain = _multi_split_gain(edge.risk, config=component_config)
        if gain <= 0.0:
            continue
        split_index = int(edge.session_index)
        candidate_gains[split_index] = max(
            float(gain), candidate_gains.get(split_index, -float("inf"))
        )
    return _select_optimal_split_indices(
        track,
        candidate_gains,
        max_splits=int(config.max_splits_per_component),
        min_observations=int(component_config.min_side_observations),
    )


def _select_optimal_split_indices(
    track: np.ndarray,
    candidate_gains: Mapping[int, float],
    *,
    max_splits: int,
    min_observations: int,
) -> tuple[int, ...]:
    """Select the maximum-gain compatible split set.

    The previous greedy planner committed to the highest-risk feasible bridge
    before considering lower-risk bridges. A central bridge can satisfy the
    fragment-length guard by itself while making the two side bridges mutually
    infeasible. Dynamic programming avoids that local optimum by optimizing the
    total selected split gain under the same guards and split-count cap.
    """

    split_indices = tuple(
        index
        for index in sorted({int(index) for index in candidate_gains})
        if 0 <= int(index) < int(track.size) - 1
    )
    if not split_indices or int(max_splits) < 1:
        return ()
    gains = {index: float(candidate_gains[index]) for index in split_indices}
    cache: dict[tuple[int, int, int], tuple[tuple[float, int], tuple[int, ...]]] = {}

    def best_from(
        fragment_start: int,
        candidate_start: int,
        remaining_splits: int,
    ) -> tuple[tuple[float, int], tuple[int, ...]]:
        key = (int(fragment_start), int(candidate_start), int(remaining_splits))
        if key in cache:
            return cache[key]

        best_score: tuple[float, int] = _IMPOSSIBLE_SCORE
        best_splits: tuple[int, ...] = ()
        if _observation_count(track, fragment_start, track.size - 1) >= int(
            min_observations
        ):
            best_score = (0.0, 0)

        if remaining_splits <= 0:
            cache[key] = (best_score, best_splits)
            return cache[key]

        for candidate_pos in range(candidate_start, len(split_indices)):
            split_index = split_indices[candidate_pos]
            if split_index < fragment_start:
                continue
            if _observation_count(track, fragment_start, split_index) < int(
                min_observations
            ):
                continue
            tail_score, tail_splits = best_from(
                split_index + 1,
                candidate_pos + 1,
                remaining_splits - 1,
            )
            if tail_score == _IMPOSSIBLE_SCORE:
                continue
            candidate_score = (
                tail_score[0] + gains[split_index],
                tail_score[1] + 1,
            )
            candidate_splits = (split_index, *tail_splits)
            if _is_better_split_plan(
                candidate_score,
                candidate_splits,
                best_score,
                best_splits,
            ):
                best_score = candidate_score
                best_splits = candidate_splits

        cache[key] = (best_score, best_splits)
        return cache[key]

    score, selected = best_from(0, 0, int(max_splits))
    if score == _IMPOSSIBLE_SCORE:
        return ()
    return selected


def _is_better_split_plan(
    candidate_score: tuple[float, int],
    candidate_splits: tuple[int, ...],
    best_score: tuple[float, int],
    best_splits: tuple[int, ...],
) -> bool:
    if candidate_score[0] > best_score[0] + 1e-12:
        return True
    if candidate_score[0] < best_score[0] - 1e-12:
        return False
    if candidate_score[1] != best_score[1]:
        return candidate_score[1] > best_score[1]
    return candidate_splits < best_splits


def _fragments_satisfy_min_observations(
    track: np.ndarray,
    split_indices: Sequence[int],
    *,
    min_observations: int,
) -> bool:
    return all(
        int(np.sum(fragment >= 0)) >= int(min_observations)
        for fragment in split_track_at_bridges(track, split_indices)
    )


def _observation_count(row: np.ndarray, start: int, stop: int) -> int:
    if int(stop) < int(start):
        return 0
    return int(np.sum(row[int(start) : int(stop) + 1] >= 0))


def _multi_split_gain(risk: float, *, config: ComponentCleanupConfig) -> float:
    return float(risk - config.split_penalty)


def _enrich_component_rows_with_split_plan(
    rows: Sequence[Mapping[str, float | int | str]],
    split_plan: Mapping[int, Sequence[int]],
    *,
    apply_splits: bool,
) -> list[dict[str, float | int | str]]:
    enriched: list[dict[str, float | int | str]] = []
    for row in rows:
        component_id = int(row["predicted_track_id"])
        split_indices = tuple(int(index) for index in split_plan.get(component_id, ()))
        updated = dict(row)
        updated["candidate_split_count"] = int(len(split_indices))
        updated["candidate_split_sessions"] = json.dumps(list(split_indices))
        updated["would_split_at_weakest_edge"] = int(bool(split_indices))
        updated["applied_split"] = int(len(split_indices) if apply_splits else 0)
        enriched.append(updated)
    return enriched


def _with_metadata(
    rows: Sequence[Mapping[str, float | int | str]],
    metadata: Mapping[str, Any],
) -> list[dict[str, float | int | str]]:
    formatted = {key: _format_metadata_value(value) for key, value in metadata.items()}
    return [{**dict(row), **formatted} for row in rows]


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be at least 1")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
