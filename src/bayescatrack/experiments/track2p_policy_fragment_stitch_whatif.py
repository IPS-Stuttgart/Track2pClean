"""What-if audit for stitching persistent complete-track FN fragments.

This diagnostic does not implement a deployable tracker.  It starts from the
frozen Track2pPolicy ComponentCleanup prediction, finds complete-track false
negatives, and simulates the minimal oracle edit that would turn each target
reference row into a complete predicted true positive.  The output is an error
budget for deciding whether a real fragment-stitching method is worth building.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_emulation_benchmark import (
    ThresholdMethod,
    emulate_track2p_tracks,
)
from bayescatrack.experiments.track2p_policy_audit import TrackEdge, track_edge_counter
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
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
    _track2p_baseline_eval,
    residual_error_rows,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_FRAGMENT_STITCH_WHATIF_METHOD = "track2p-policy-fragment-stitch-whatif"
CompleteTrack = tuple[int, ...]


@dataclass(frozen=True)
class FragmentStitchWhatIfResult:
    """Detailed candidate rows plus a ranked summary."""

    rows: tuple[dict[str, float | int | str], ...]
    summary_rows: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True)
class _SubjectState:
    subject: str
    cleaned_full: np.ndarray
    cleaned_eval: np.ndarray
    reference_tracks: np.ndarray
    reference: Any
    baseline_scores: Mapping[str, float | int]
    reference_eval: np.ndarray
    policy_eval: np.ndarray
    track2p_eval: np.ndarray
    gap_eval: np.ndarray


@dataclass(frozen=True)
class _RepairPlan:
    selected_rows: tuple[int, ...]
    sessions_present: tuple[int, ...]
    edge_additions: int
    edge_swaps: int
    component_merges: int
    existing_correct_edges: tuple[TrackEdge, ...]
    missing_edges: tuple[TrackEdge, ...]
    wrong_edges: tuple[TrackEdge, ...]
    structural_risk: str

    @property
    def edit_count(self) -> int:
        return int(self.edge_additions + self.edge_swaps + self.component_merges)


def run_track2p_policy_fragment_stitch_whatif(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
) -> FragmentStitchWhatIfResult:
    """Return one oracle stitch what-if row per complete-track FN."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
    )
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    cleanup_config = cleanup_config or ComponentCleanupConfig()
    states: list[_SubjectState] = []
    for subject_dir in subject_dirs:
        states.append(
            _subject_state(
                subject_dir,
                config=policy_config,
                cleanup_config=cleanup_config,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
            )
        )
    baseline_global = _global_counts(state.baseline_scores for state in states)

    rows: list[dict[str, float | int | str]] = []
    for state in states:
        residual_rows = residual_error_rows(
            state.cleaned_eval,
            state.reference_eval,
            subject=state.subject,
            track2p_tracks=state.track2p_eval,
            policy_tracks=state.policy_eval,
            gap_tracks=state.gap_eval,
            seed_session=policy_config.seed_session,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            cell_probability_threshold=float(policy_config.cell_probability_threshold),
            transform_type=str(policy_config.transform_type),
        )
        complete_fns = _complete_fn_tracks(residual_rows)
        for track in complete_fns:
            rows.append(
                _whatif_row(
                    state,
                    track,
                    baseline_global=baseline_global,
                    config=policy_config,
                    threshold_method=threshold_method,
                    iou_distance_threshold=float(iou_distance_threshold),
                )
            )

    summary_rows = tuple(sorted(rows, key=_summary_rank_key))
    return FragmentStitchWhatIfResult(tuple(rows), summary_rows)


def _subject_state(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> _SubjectState:
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError(
            "Track2p-policy fragment-stitch what-if requires independent manual GT references"
        )
    sessions = _load_subject_sessions(subject_dir, config)
    _validate_reference_roi_indices(reference, sessions)
    reference_tracks = _reference_matrix(reference, curated_only=config.curated_only)
    policy_prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    policy_full = _normalize_int_track_matrix(policy_prediction.tracks)
    policy_eval, reference_eval, evaluated_track_ids = _evaluated_prediction_rows(
        policy_full, reference_tracks, config=config
    )
    audit_rows = component_audit_rows(
        policy_eval,
        reference_eval,
        sessions=sessions,
        diagnostics=cast(
            Sequence[Track2pPolicyLinkDiagnostic], policy_prediction.diagnostics
        ),
        subject=subject_dir.name,
        config=cleanup_config,
        track_ids=evaluated_track_ids,
        seed_session=config.seed_session,
    )
    cleaned_full = apply_weakest_bridge_splits(
        policy_full, _mark_applied_splits(audit_rows, apply_splits=True)
    )
    baseline_scores = _score_prediction_against_reference(
        cleaned_full, reference, config=config
    )
    cleaned_eval, reference_eval, _ = _evaluated_prediction_rows(
        cleaned_full, reference_tracks, config=config
    )
    track2p_eval = _track2p_baseline_eval(subject_dir, reference_tracks, config=config)
    gap_full = _normalize_int_track_matrix(
        emulate_track2p_tracks(
            sessions,
            transform_type=config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            max_gap=int(config.max_gap),
        )
    )
    gap_eval, _, _ = _evaluated_prediction_rows(
        gap_full, reference_tracks, config=config
    )
    return _SubjectState(
        subject=subject_dir.name,
        cleaned_full=cleaned_full,
        cleaned_eval=cleaned_eval,
        reference_tracks=reference_tracks,
        reference=reference,
        baseline_scores=baseline_scores,
        reference_eval=reference_eval,
        policy_eval=policy_eval,
        track2p_eval=track2p_eval,
        gap_eval=gap_eval,
    )


def _whatif_row(
    state: _SubjectState,
    track: CompleteTrack,
    *,
    baseline_global: Mapping[str, int],
    config: Track2pBenchmarkConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> dict[str, float | int | str]:
    plan = _minimal_repair_plan(state.cleaned_full, track)
    candidate = _apply_repair_plan(state.cleaned_full, track, plan)
    candidate_scores = _score_prediction_against_reference(
        candidate, state.reference, config=config
    )
    subject_delta = _score_delta(state.baseline_scores, candidate_scores)
    global_after = _apply_delta_to_global_counts(baseline_global, subject_delta)
    duplicate_source, duplicate_target = _duplicate_flags(
        state.cleaned_full, track, plan.selected_rows
    )
    reference_edges = _track_edges(track)
    missing_edge_counts = Counter(plan.missing_edges)
    track2p_counts = track_edge_counter(state.track2p_eval)
    policy_counts = track_edge_counter(state.policy_eval)
    gap_counts = track_edge_counter(state.gap_eval)
    track2p_supported = tuple(
        edge for edge in plan.missing_edges if track2p_counts.get(edge, 0) > 0
    )
    policy_supported = tuple(
        edge for edge in plan.missing_edges if policy_counts.get(edge, 0) > 0
    )
    gap_supported = tuple(
        edge for edge in plan.missing_edges if gap_counts.get(edge, 0) > 0
    )
    selected_complete_fp = _selected_complete_fp_count(
        state.cleaned_full, plan, state.reference_tracks
    )
    return {
        "subject": state.subject,
        "reference_track_id": _track_id(track),
        "candidate_repair": _candidate_repair(plan),
        "sessions_present": _session_list(plan.sessions_present),
        "predicted_fragment_count": int(len(plan.selected_rows)),
        "fragment_session_spans": _fragment_session_spans(
            state.cleaned_full, track, plan.selected_rows
        ),
        "missing_adjacent_edges": _edge_list(plan.missing_edges),
        "wrong_adjacent_edges": _edge_list(plan.wrong_edges),
        "track2p_supported_missing_edges": _edge_list(track2p_supported),
        "policy_supported_missing_edges": _edge_list(policy_supported),
        "gap_supported_missing_edges": _edge_list(gap_supported),
        "would_require_edge_addition_count": int(plan.edge_additions),
        "would_require_edge_swap_count": int(plan.edge_swaps),
        "would_require_component_merge_count": int(plan.component_merges),
        "minimal_edit_count": int(plan.edit_count),
        "would_create_duplicate_source": int(duplicate_source),
        "would_create_duplicate_target": int(duplicate_target),
        "would_merge_into_complete_fp": int(selected_complete_fp > 0),
        "selected_complete_fp_count": int(selected_complete_fp),
        "pairwise_tp_delta": int(subject_delta["pairwise_true_positives"]),
        "pairwise_fp_delta": int(subject_delta["pairwise_false_positives"]),
        "pairwise_fn_delta": int(subject_delta["pairwise_false_negatives"]),
        "complete_tp_delta": int(subject_delta["complete_track_true_positives"]),
        "complete_fp_delta": int(subject_delta["complete_track_false_positives"]),
        "complete_fn_delta": int(subject_delta["complete_track_false_negatives"]),
        "new_pairwise_f1_micro": _f1_from_counts(
            global_after["pairwise_true_positives"],
            global_after["pairwise_false_positives"],
            global_after["pairwise_false_negatives"],
        ),
        "new_complete_track_f1_micro": _f1_from_counts(
            global_after["complete_track_true_positives"],
            global_after["complete_track_false_positives"],
            global_after["complete_track_false_negatives"],
        ),
        "expected_complete_track_f1_delta": (
            _f1_from_counts(
                global_after["complete_track_true_positives"],
                global_after["complete_track_false_positives"],
                global_after["complete_track_false_negatives"],
            )
            - _f1_from_counts(
                baseline_global["complete_track_true_positives"],
                baseline_global["complete_track_false_positives"],
                baseline_global["complete_track_false_negatives"],
            )
        ),
        "expected_pairwise_f1_delta": (
            _f1_from_counts(
                global_after["pairwise_true_positives"],
                global_after["pairwise_false_positives"],
                global_after["pairwise_false_negatives"],
            )
            - _f1_from_counts(
                baseline_global["pairwise_true_positives"],
                baseline_global["pairwise_false_positives"],
                baseline_global["pairwise_false_negatives"],
            )
        ),
        "structural_risk": _structural_risk(
            plan,
            duplicate_source=duplicate_source,
            duplicate_target=duplicate_target,
            complete_fp_delta=int(subject_delta["complete_track_false_positives"]),
        ),
        "reference_adjacent_edges": _edge_list(reference_edges),
        "missing_adjacent_edge_count": int(sum(missing_edge_counts.values())),
        "track2p_supported_missing_edge_count": int(len(track2p_supported)),
        "policy_supported_missing_edge_count": int(len(policy_supported)),
        "gap_supported_missing_edge_count": int(len(gap_supported)),
        "threshold_method": str(threshold_method),
        "iou_distance_threshold": float(iou_distance_threshold),
        "cell_probability_threshold": float(config.cell_probability_threshold),
        "transform_type": str(config.transform_type),
    }


def _minimal_repair_plan(predicted: np.ndarray, track: CompleteTrack) -> _RepairPlan:
    candidates = _overlap_rows(predicted, track)
    subsets = _candidate_subsets(candidates)
    if not subsets:
        return _repair_plan_for_rows(predicted, track, ())
    return min(
        (_repair_plan_for_rows(predicted, track, subset) for subset in subsets),
        key=lambda plan: (
            plan.edit_count,
            _risk_rank(plan.structural_risk),
            -len(plan.sessions_present),
            len(plan.selected_rows),
        ),
    )


def _repair_plan_for_rows(
    predicted: np.ndarray, track: CompleteTrack, rows: Sequence[int]
) -> _RepairPlan:
    selected_rows = tuple(int(row) for row in rows)
    reference_edges = _track_edges(track)
    existing_correct_edges = tuple(
        edge
        for row_index in selected_rows
        for edge in _row_edges(predicted[row_index])
        if edge in reference_edges
    )
    wrong_edges = tuple(
        edge
        for row_index in selected_rows
        for edge in _row_edges(predicted[row_index])
        if _edge_touches_track(edge, track) and edge not in reference_edges
    )
    sessions_present = tuple(
        sorted(
            {
                session
                for row_index in selected_rows
                for session, roi in enumerate(predicted[row_index])
                if session < len(track) and int(roi) == int(track[session])
            }
        )
    )
    missing_edges = tuple(
        edge for edge in reference_edges if edge not in existing_correct_edges
    )
    swaps = _swap_count(predicted, track, selected_rows)
    additions = len(missing_edges) if selected_rows else len(reference_edges)
    merges = max(0, len(selected_rows) - 1)
    return _RepairPlan(
        selected_rows=selected_rows,
        sessions_present=sessions_present,
        edge_additions=additions,
        edge_swaps=swaps,
        component_merges=merges,
        existing_correct_edges=existing_correct_edges,
        missing_edges=missing_edges,
        wrong_edges=wrong_edges,
        structural_risk="low",
    )


def _apply_repair_plan(
    predicted: np.ndarray, track: CompleteTrack, plan: _RepairPlan
) -> np.ndarray:
    target = np.asarray(track, dtype=int)
    if not plan.selected_rows:
        return np.vstack([predicted, target.reshape(1, -1)]).astype(int, copy=False)
    output = np.asarray(predicted, dtype=int).copy()
    keep_row = int(plan.selected_rows[0])
    output[keep_row] = target
    drop = set(int(row) for row in plan.selected_rows[1:])
    if drop:
        keep_indices = [index for index in range(output.shape[0]) if index not in drop]
        output = output[np.asarray(keep_indices, dtype=int)]
    return output


def _complete_fn_tracks(
    rows: Sequence[Mapping[str, float | int | str]],
) -> tuple[CompleteTrack, ...]:
    return tuple(
        _parse_track_id(str(row["track_id_or_edge"]))
        for row in rows
        if str(row.get("error_type")) == "complete_fn"
    )


def _overlap_rows(predicted: np.ndarray, track: CompleteTrack) -> tuple[int, ...]:
    rows: list[int] = []
    for row_index, row in enumerate(predicted):
        if any(
            session < len(row) and int(row[session]) == int(roi)
            for session, roi in enumerate(track)
        ):
            rows.append(int(row_index))
    return tuple(rows)


def _candidate_subsets(rows: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    if not rows:
        return ()
    limited = tuple(rows[:12])
    subsets: list[tuple[int, ...]] = []
    for mask in range(1, 1 << len(limited)):
        subsets.append(
            tuple(
                limited[index] for index in range(len(limited)) if mask & (1 << index)
            )
        )
    return tuple(subsets)


def _swap_count(
    predicted: np.ndarray, track: CompleteTrack, selected_rows: Sequence[int]
) -> int:
    swaps = 0
    for row_index in selected_rows:
        row = predicted[int(row_index)]
        for session, reference_roi in enumerate(track):
            if session >= row.size:
                continue
            value = int(row[session])
            if value >= 0 and value != int(reference_roi):
                swaps += 1
    return int(swaps)


def _duplicate_flags(
    predicted: np.ndarray, track: CompleteTrack, selected_rows: Sequence[int]
) -> tuple[bool, bool]:
    selected = set(int(row) for row in selected_rows)
    duplicate_source = False
    duplicate_target = False
    reference_edges = _track_edges(track)
    for row_index, row in enumerate(predicted):
        if row_index in selected:
            continue
        for edge in _row_edges(row):
            for ref_edge in reference_edges:
                if edge[0] != ref_edge[0] or edge[1] != ref_edge[1]:
                    continue
                duplicate_source = duplicate_source or (
                    edge[2] == ref_edge[2] and edge[3] != ref_edge[3]
                )
                duplicate_target = duplicate_target or (
                    edge[2] != ref_edge[2] and edge[3] == ref_edge[3]
                )
    return duplicate_source, duplicate_target


def _selected_complete_fp_count(
    predicted: np.ndarray, plan: _RepairPlan, reference: np.ndarray
) -> int:
    reference_counter = Counter(tuple(int(value) for value in row) for row in reference)
    count = 0
    for row_index in plan.selected_rows:
        row = predicted[int(row_index)]
        row_key = tuple(int(value) for value in row)
        if np.all(row >= 0) and reference_counter.get(row_key, 0) <= 0:
            count += 1
    return count


def _global_counts(score_rows: Sequence[Mapping[str, float | int]]) -> dict[str, int]:
    materialized_rows = tuple(score_rows)
    keys = (
        "pairwise_true_positives",
        "pairwise_false_positives",
        "pairwise_false_negatives",
        "complete_track_true_positives",
        "complete_track_false_positives",
        "complete_track_false_negatives",
    )
    return {key: int(sum(int(row[key]) for row in materialized_rows)) for key in keys}


def _score_delta(
    baseline: Mapping[str, float | int],
    candidate: Mapping[str, float | int],
) -> dict[str, int]:
    return {
        key: int(candidate[key]) - int(baseline[key])
        for key in (
            "pairwise_true_positives",
            "pairwise_false_positives",
            "pairwise_false_negatives",
            "complete_track_true_positives",
            "complete_track_false_positives",
            "complete_track_false_negatives",
        )
    }


def _apply_delta_to_global_counts(
    baseline_global: Mapping[str, int], delta: Mapping[str, int]
) -> dict[str, int]:
    return {
        key: int(baseline_global[key]) + int(delta.get(key, 0))
        for key in baseline_global
    }


def _track_edges(track: CompleteTrack) -> tuple[TrackEdge, ...]:
    return tuple(
        (index, index + 1, int(track[index]), int(track[index + 1]))
        for index in range(max(0, len(track) - 1))
        if int(track[index]) >= 0 and int(track[index + 1]) >= 0
    )


def _row_edges(row: np.ndarray) -> tuple[TrackEdge, ...]:
    return tuple(
        (index, index + 1, int(row[index]), int(row[index + 1]))
        for index in range(max(0, row.size - 1))
        if row[index] >= 0 and row[index + 1] >= 0
    )


def _edge_touches_track(edge: TrackEdge, track: CompleteTrack) -> bool:
    session_a, session_b, roi_a, roi_b = edge
    return bool(
        (session_a < len(track) and int(track[session_a]) == int(roi_a))
        or (session_b < len(track) and int(track[session_b]) == int(roi_b))
    )


def _structural_risk(
    plan: _RepairPlan,
    *,
    duplicate_source: bool,
    duplicate_target: bool,
    complete_fp_delta: int,
) -> str:
    if duplicate_source or duplicate_target or complete_fp_delta > 0:
        return "high"
    if plan.edge_swaps > 0 or plan.component_merges > 0:
        return "medium"
    return "low"


def _risk_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value, 3)


def _summary_rank_key(row: Mapping[str, float | int | str]) -> tuple[float, int, float]:
    return (
        -float(row["expected_complete_track_f1_delta"]),
        int(row["minimal_edit_count"]),
        -float(row["expected_pairwise_f1_delta"]),
    )


def _parse_track_id(value: str) -> CompleteTrack:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _track_id(track: CompleteTrack) -> str:
    return ",".join(str(value) for value in track)


def _edge_id(edge: TrackEdge) -> str:
    session_a, session_b, roi_a, roi_b = edge
    return f"{session_a}:{roi_a}->{session_b}:{roi_b}"


def _candidate_repair(plan: _RepairPlan) -> str:
    parts: list[str] = []
    if plan.component_merges:
        parts.append(f"merge {plan.component_merges + 1} fragments")
    if plan.edge_additions:
        parts.append(f"add {plan.edge_additions} missing adjacent edges")
    if plan.edge_swaps:
        parts.append(f"swap {plan.edge_swaps} occupied ROI slots")
    if not parts:
        return "already complete"
    return "; ".join(parts)


def _edge_list(edges: Sequence[TrackEdge]) -> str:
    return ";".join(_edge_id(edge) for edge in edges)


def _session_list(sessions: Sequence[int]) -> str:
    return ",".join(str(session) for session in sessions)


def _fragment_session_spans(
    predicted: np.ndarray, track: CompleteTrack, selected_rows: Sequence[int]
) -> str:
    spans: list[str] = []
    for row_index in selected_rows:
        row = predicted[int(row_index)]
        sessions = [
            session
            for session, roi in enumerate(track)
            if session < row.size and int(row[session]) == int(roi)
        ]
        if not sessions:
            continue
        spans.append(f"{int(row_index)}:{min(sessions)}-{max(sessions)}")
    return ";".join(spans)


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)
    if denominator == 0:
        return 1.0
    return float(2 * int(tp) / denominator)


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
    """Build the command-line parser for fragment-stitch what-if audit."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-fragment-stitch-whatif",
        description=(
            "Simulate minimal oracle fragment-stitch edits for complete-track false negatives after Track2p-policy ComponentCleanup."
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
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fragment-stitch what-if CLI."""

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
        seed_session=args.seed_session,
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
    result = run_track2p_policy_fragment_stitch_whatif(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
    )
    write_rows(result.rows, args.output, output_format=args.format)
    if args.summary_output is not None:
        write_rows(result.summary_rows, args.summary_output, output_format=args.format)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
