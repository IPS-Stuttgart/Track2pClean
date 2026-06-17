"""Rank teacher-free adjacent rescue candidates after the non-teacher lead row.

This audit starts from ``CoherenceSuffixStitch + GrowthVeto`` by default and
asks whether the adjacent edits previously supplied by Track2p-teacher rescue
are recoverable from BayesCaTrack-only features.  Track2p support and manual-GT
status are emitted only as audit labels after ranking; they are not inputs to
the ranking score.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_stitch_whatif as suffix,
)
from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup
from bayescatrack.experiments import track2p_policy_growth_veto_whatif as veto
from bayescatrack.experiments import track2p_policy_suffix_stitch_ranking_audit as rank
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_audit import TrackEdge, track_edge_counter
from bayescatrack.experiments.track2p_policy_benchmark import track2p_policy_config
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
    _try_apply_teacher_edge,
)

METHOD = "track2p-policy-teacher-free-adjacent-rescue-ranking-audit"
BasePrediction = Literal["coherence-suffix-growth-veto", "coherence-suffix"]
TARGET_TEACHER_EDGES: tuple[TrackEdge, ...] = (
    (3, 4, 643, 624),
    (4, 5, 624, 616),
)


@dataclass(frozen=True)
class TeacherFreeAdjacentRescueRankingAuditResult:
    """Teacher-free adjacent-rescue candidate and summary rows."""

    rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


def run_track2p_policy_teacher_free_adjacent_rescue_ranking_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = "min",
    iou_distance_threshold: float = 12.0,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    suffix_gate: suffix.CoherenceSuffixStitchGate | None = None,
    growth_veto_gate: cleanup.GrowthVetoGate | None = None,
    base_prediction: BasePrediction = "coherence-suffix-growth-veto",
    edge_top_k: int = 25,
    path_beam_width: int = 100,
    anchor_min_registered_iou: float = 0.50,
    anchor_min_shifted_iou: float = 0.30,
    anchor_min_cell_probability: float = 0.80,
    progress: bool = False,
) -> TeacherFreeAdjacentRescueRankingAuditResult:
    """Rank conflict-free adjacent continuation candidates from non-GT features."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    suffix_gate = suffix_gate or suffix.CoherenceSuffixStitchGate()
    growth_veto_gate = growth_veto_gate or cleanup.GrowthVetoGate(
        max_local_neighbor_distortion=None
    )
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    all_rows: list[dict[str, Any]] = []
    for subject_dir in subject_dirs:
        all_rows.extend(
            _subject_rows(
                subject_dir,
                config=policy_config,
                cleanup_config=cleanup_config,
                suffix_gate=suffix_gate,
                growth_veto_gate=growth_veto_gate,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                base_prediction=base_prediction,
                edge_top_k=int(edge_top_k),
                path_beam_width=int(path_beam_width),
                anchor_min_registered_iou=float(anchor_min_registered_iou),
                anchor_min_shifted_iou=float(anchor_min_shifted_iou),
                anchor_min_cell_probability=float(anchor_min_cell_probability),
                progress=progress,
            )
        )
    _assign_ranks(all_rows)
    return TeacherFreeAdjacentRescueRankingAuditResult(
        tuple(all_rows), tuple(_summary_rows(all_rows))
    )


def _subject_rows(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    suffix_gate: suffix.CoherenceSuffixStitchGate,
    growth_veto_gate: cleanup.GrowthVetoGate,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    base_prediction: BasePrediction,
    edge_top_k: int,
    path_beam_width: int,
    anchor_min_registered_iou: float,
    anchor_min_shifted_iou: float,
    anchor_min_cell_probability: float,
    progress: bool,
) -> list[dict[str, Any]]:
    state = veto._subject_state(
        subject_dir,
        config=config,
        cleanup_config=cleanup_config,
        suffix_gate=suffix_gate,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        edge_top_k=int(edge_top_k),
        path_beam_width=int(path_beam_width),
        anchor_min_registered_iou=float(anchor_min_registered_iou),
        anchor_min_shifted_iou=float(anchor_min_shifted_iou),
        anchor_min_cell_probability=float(anchor_min_cell_probability),
        prediction_base="coherence-suffix",
        progress=progress,
    )
    predicted = state.combined
    if base_prediction == "coherence-suffix-growth-veto":
        predicted = _apply_growth_veto_base(
            state,
            gate=growth_veto_gate,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            cell_probability_threshold=float(config.cell_probability_threshold),
            transform_type=str(config.transform_type),
        )
    elif base_prediction != "coherence-suffix":
        raise ValueError(f"Unsupported base prediction: {base_prediction!r}")

    feature_cache = rank._FeatureCache(
        sessions=state.sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(config.cell_probability_threshold),
        matrices={},
    )
    baseline_scores = dict(score_track_matrices(predicted, state.reference))
    reference_counts = track_edge_counter(state.reference)
    predicted_counts = track_edge_counter(predicted)
    teacher_counts = track_edge_counter(state.teacher)
    candidates = _adjacent_candidates(
        predicted,
        feature_cache=feature_cache,
        edge_top_k=int(edge_top_k),
    )
    candidate_paths = _sequential_target_extension_candidate_paths(
        predicted,
        feature_cache=feature_cache,
        edge_top_k=int(edge_top_k),
    )
    rows: list[dict[str, Any]] = []
    for action, edge_candidate in candidates:
        edge = edge_candidate.edge
        if predicted_counts.get(edge, 0) > 0:
            continue
        candidate_tracks, attempt = _try_apply_teacher_edge(
            predicted,
            edge,
            seed_session=int(config.seed_session),
            allow_completing_rescue=False,
            allow_source_backfill=True,
            allow_fragment_merges=True,
            min_component_observations=1,
        )
        if int(attempt.get("applied", 0)) <= 0:
            continue
        candidate_scores = dict(score_track_matrices(candidate_tracks, state.reference))
        delta = suffix._score_delta(baseline_scores, candidate_scores)
        growth = veto._edge_growth_features_fast(
            state.growth_context,
            edge,
            model=state.growth_models.get(
                (edge[0], edge[1]), veto._identity_growth_model()
            ),
        )
        row = _candidate_row(
            subject=state.subject,
            action=action,
            edge_candidate=edge_candidate,
            growth=growth,
            predicted=predicted,
            attempt=attempt,
            reference_counts=reference_counts,
            teacher_counts=teacher_counts,
            delta=delta,
        )
        rows.append(row)
    for path_index, path_edges in enumerate(candidate_paths, start=1):
        candidate_tracks, attempts = _try_apply_edge_path(
            predicted,
            tuple(edge.edge for edge in path_edges),
            seed_session=int(config.seed_session),
        )
        if any(int(attempt.get("applied", 0)) <= 0 for attempt in attempts):
            continue
        candidate_scores = dict(score_track_matrices(candidate_tracks, state.reference))
        delta = suffix._score_delta(baseline_scores, candidate_scores)
        path_score = _path_ranking_score(
            path_edges, state.growth_context, state.growth_models
        )
        for step_index, (edge_candidate, attempt) in enumerate(
            zip(path_edges, attempts, strict=True), start=1
        ):
            edge = edge_candidate.edge
            growth = veto._edge_growth_features_fast(
                state.growth_context,
                edge,
                model=state.growth_models.get(
                    (edge[0], edge[1]), veto._identity_growth_model()
                ),
            )
            row = _candidate_row(
                subject=state.subject,
                action=f"two_edge_target_extension_step_{step_index}",
                edge_candidate=edge_candidate,
                growth=growth,
                predicted=predicted,
                attempt=attempt,
                reference_counts=reference_counts,
                teacher_counts=teacher_counts,
                delta=delta,
            )
            row["candidate_path"] = _edge_list(tuple(edge.edge for edge in path_edges))
            row["path_index"] = int(path_index)
            row["path_step"] = int(step_index)
            row["path_length"] = int(len(path_edges))
            row["path_ranking_score"] = float(path_score)
            row["ranking_score"] = float(path_score)
            rows.append(row)
    return rows


def teacher_free_adjacent_ranking_score(
    *,
    registered_iou: float,
    shifted_iou: float,
    roi_aware_score: float,
    centroid_distance: float,
    area_ratio: float,
    cell_probability_a: float,
    cell_probability_b: float,
    row_rank: int,
    column_rank: int,
    row_margin: float,
    column_margin: float,
    threshold_margin: float,
    activity_similarity: float,
    growth_residual_mahalanobis: float,
    two_edge_motion_consistency: float,
    would_complete_predicted_row: int,
    would_merge_components: int,
) -> float:
    """Score a candidate using only non-GT, non-Track2p features."""

    min_probability = _nanmin_default((cell_probability_a, cell_probability_b), 0.5)
    local_rank_bonus = 0.10 / max(1, int(row_rank)) + 0.10 / max(1, int(column_rank))
    motion = _finite_or(two_edge_motion_consistency, 0.5)
    growth_penalty = min(_finite_or(growth_residual_mahalanobis, 10.0), 30.0) / 300.0
    return float(
        0.60 * _finite_or(shifted_iou, 0.0)
        + 0.35 * _finite_or(registered_iou, 0.0)
        + 0.20 * _finite_or(roi_aware_score, 0.0)
        + 0.08 * _finite_or(area_ratio, 0.0)
        + 0.08 * min_probability
        + 0.06 * max(0.0, _finite_or(row_margin, 0.0))
        + 0.06 * max(0.0, _finite_or(column_margin, 0.0))
        + 0.03 * max(0.0, _finite_or(threshold_margin, 0.0))
        + 0.04 * _finite_or(activity_similarity, 0.5)
        + 0.08 * motion
        + local_rank_bonus
        + 0.12 * int(would_complete_predicted_row)
        + 0.06 * int(would_merge_components)
        - 0.012 * _finite_or(centroid_distance, 30.0)
        - growth_penalty
    )


def _apply_growth_veto_base(
    state: veto._SubjectState,
    *,
    gate: cleanup.GrowthVetoGate,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cell_probability_threshold: float,
    transform_type: str,
) -> np.ndarray:
    baseline_scores = dict(score_track_matrices(state.combined, state.reference))
    edge_rows = veto._accepted_edge_rows(
        state,
        global_baseline_scores=baseline_scores,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(cell_probability_threshold),
        transform_type=str(transform_type),
    )
    edge_rows = cleanup._augment_growth_veto_candidate_shifted_iou(
        edge_rows,
        state.sessions,
        gate=gate,
        n_sessions=int(state.reference.shape[1]),
    )
    selected = cleanup._selected_growth_veto_rows(
        edge_rows, gate=gate, n_sessions=int(state.reference.shape[1])
    )
    vetoed, _applied = cleanup._apply_growth_veto_rows(
        state.combined, selected, gate=gate
    )
    return vetoed


def _adjacent_candidates(
    predicted: np.ndarray,
    *,
    feature_cache: rank._FeatureCache,
    edge_top_k: int,
) -> tuple[tuple[str, rank._EdgeCandidate], ...]:
    seen: set[TrackEdge] = set()
    rows: list[tuple[str, rank._EdgeCandidate]] = []
    for row in np.asarray(predicted, dtype=int):
        for session_index in range(max(0, row.size - 1)):
            source_roi = int(row[session_index])
            target_roi = int(row[session_index + 1])
            if source_roi >= 0 and target_roi < 0:
                for edge in rank._top_edge_candidates(
                    feature_cache, session_index, source_roi, top_k=int(edge_top_k)
                ):
                    if edge.edge not in seen:
                        seen.add(edge.edge)
                        rows.append(("target_extension", edge))
            if source_roi < 0 and target_roi >= 0:
                for edge in _top_reverse_edge_candidates(
                    feature_cache, session_index, target_roi, top_k=int(edge_top_k)
                ):
                    if edge.edge not in seen:
                        seen.add(edge.edge)
                        rows.append(("source_backfill", edge))
    return tuple(rows)


def _sequential_target_extension_candidate_paths(
    predicted: np.ndarray,
    *,
    feature_cache: rank._FeatureCache,
    edge_top_k: int,
) -> tuple[tuple[rank._EdgeCandidate, ...], ...]:
    output: list[tuple[rank._EdgeCandidate, ...]] = []
    seen: set[tuple[TrackEdge, ...]] = set()
    top_k = max(1, min(int(edge_top_k), 25))
    for row in np.asarray(predicted, dtype=int):
        for session_index in range(max(0, row.size - 2)):
            if int(row[session_index]) < 0:
                continue
            if int(row[session_index + 1]) >= 0:
                continue
            if int(row[session_index + 2]) >= 0:
                continue
            first_edges = rank._top_edge_candidates(
                feature_cache,
                session_index,
                int(row[session_index]),
                top_k=top_k,
            )
            for first in first_edges:
                second_edges = rank._top_edge_candidates(
                    feature_cache,
                    session_index + 1,
                    int(first.edge[3]),
                    top_k=top_k,
                )
                for second in second_edges:
                    path = (first.edge, second.edge)
                    if path in seen:
                        continue
                    seen.add(path)
                    output.append((first, second))
    output.sort(key=lambda edges: _mean_ranking_score(edges), reverse=True)
    return tuple(output)


def _try_apply_edge_path(
    predicted: np.ndarray, edges: Sequence[TrackEdge], *, seed_session: int
) -> tuple[np.ndarray, tuple[dict[str, int | str], ...]]:
    output = np.asarray(predicted, dtype=int).copy()
    attempts: list[dict[str, int | str]] = []
    for edge in edges:
        output, attempt = _try_apply_teacher_edge(
            output,
            edge,
            seed_session=int(seed_session),
            allow_completing_rescue=False,
            allow_source_backfill=True,
            allow_fragment_merges=True,
            min_component_observations=1,
        )
        attempts.append(attempt)
        if int(attempt.get("applied", 0)) <= 0:
            break
    return output, tuple(attempts)


def _path_ranking_score(
    edges: Sequence[rank._EdgeCandidate],
    growth_context: veto._GrowthFeatureContext,
    growth_models: Mapping[tuple[int, int], Any],
) -> float:
    if not edges:
        return float("-inf")
    edge_scores = []
    for edge_candidate in edges:
        edge = edge_candidate.edge
        growth = veto._edge_growth_features_fast(
            growth_context,
            edge,
            model=growth_models.get((edge[0], edge[1]), veto._identity_growth_model()),
        )
        edge_scores.append(
            teacher_free_adjacent_ranking_score(
                registered_iou=edge_candidate.registered_iou,
                shifted_iou=edge_candidate.shifted_iou,
                roi_aware_score=edge_candidate.roi_aware_score,
                centroid_distance=edge_candidate.centroid_distance,
                area_ratio=edge_candidate.area_ratio,
                cell_probability_a=edge_candidate.cell_probability_a,
                cell_probability_b=edge_candidate.cell_probability_b,
                row_rank=edge_candidate.row_rank,
                column_rank=edge_candidate.column_rank,
                row_margin=edge_candidate.row_margin,
                column_margin=edge_candidate.column_margin,
                threshold_margin=edge_candidate.threshold_margin,
                activity_similarity=edge_candidate.activity_similarity,
                growth_residual_mahalanobis=float(growth.growth_residual_mahalanobis),
                two_edge_motion_consistency=float(growth.two_edge_motion_consistency),
                would_complete_predicted_row=0,
                would_merge_components=0,
            )
        )
    return float(
        float(np.mean(np.asarray(edge_scores, dtype=float)))
        + 0.15 * rank._motion_consistency(edges)
        + 0.04 * len(edges)
    )


def _mean_ranking_score(edges: Sequence[rank._EdgeCandidate]) -> float:
    if not edges:
        return float("-inf")
    return float(np.mean([float(edge.edge_score) for edge in edges]))


def _top_reverse_edge_candidates(
    feature_cache: rank._FeatureCache,
    session_index: int,
    target_roi: int,
    *,
    top_k: int,
) -> tuple[rank._EdgeCandidate, ...]:
    if session_index < 0 or session_index + 1 >= len(feature_cache.sessions):
        return ()
    matrices = feature_cache.pair(session_index)
    target_matches = np.flatnonzero(matrices.target_indices == int(target_roi))
    if target_matches.size == 0:
        return ()
    target_local = int(target_matches[0])
    candidates: list[rank._EdgeCandidate] = []
    for source_local, source_roi in enumerate(matrices.source_indices):
        cell_probability_a = rank._cell_probability(
            feature_cache.sessions, session_index, int(source_roi)
        )
        if np.isfinite(cell_probability_a) and (
            cell_probability_a < feature_cache.cell_probability_threshold
        ):
            continue
        registered_iou = float(matrices.registered_iou[source_local, target_local])
        shifted_iou = float(matrices.shifted_iou[source_local, target_local])
        if max(registered_iou, shifted_iou) <= 0.0:
            continue
        candidates.append(
            rank._edge_candidate(
                feature_cache,
                matrices,
                session_index=session_index,
                source_local=int(source_local),
                target_local=target_local,
            )
        )
    candidates.sort(key=lambda edge: edge.edge_score, reverse=True)
    return tuple(candidates[: int(top_k)])


def _candidate_row(
    *,
    subject: str,
    action: str,
    edge_candidate: rank._EdgeCandidate,
    growth: Any,
    predicted: np.ndarray,
    attempt: Mapping[str, Any],
    reference_counts: Counter[TrackEdge],
    teacher_counts: Counter[TrackEdge],
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    edge = edge_candidate.edge
    source_row = int(attempt.get("source_row", -1))
    target_row = int(attempt.get("target_row", -1))
    source_observations = _row_observations(predicted, source_row)
    target_observations = _row_observations(predicted, target_row)
    would_complete = int(
        _would_complete_predicted_row(predicted, source_row, target_row, edge)
    )
    would_merge = int(source_row >= 0 and target_row >= 0 and source_row != target_row)
    score = teacher_free_adjacent_ranking_score(
        registered_iou=edge_candidate.registered_iou,
        shifted_iou=edge_candidate.shifted_iou,
        roi_aware_score=edge_candidate.roi_aware_score,
        centroid_distance=edge_candidate.centroid_distance,
        area_ratio=edge_candidate.area_ratio,
        cell_probability_a=edge_candidate.cell_probability_a,
        cell_probability_b=edge_candidate.cell_probability_b,
        row_rank=edge_candidate.row_rank,
        column_rank=edge_candidate.column_rank,
        row_margin=edge_candidate.row_margin,
        column_margin=edge_candidate.column_margin,
        threshold_margin=edge_candidate.threshold_margin,
        activity_similarity=edge_candidate.activity_similarity,
        growth_residual_mahalanobis=float(growth.growth_residual_mahalanobis),
        two_edge_motion_consistency=float(growth.two_edge_motion_consistency),
        would_complete_predicted_row=would_complete,
        would_merge_components=would_merge,
    )
    return {
        "subject": subject,
        "session_a": int(edge[0]),
        "session_b": int(edge[1]),
        "roi_a": int(edge[2]),
        "roi_b": int(edge[3]),
        "candidate_action": action,
        "candidate_reason": str(attempt.get("reason", "")),
        "source_row": source_row,
        "target_row": target_row,
        "source_chain_length": source_observations,
        "target_chain_length": target_observations,
        "would_merge_components": would_merge,
        "would_complete_predicted_row": would_complete,
        "registered_iou": float(edge_candidate.registered_iou),
        "shifted_iou": float(edge_candidate.shifted_iou),
        "roi_aware_score": float(edge_candidate.roi_aware_score),
        "centroid_distance": float(edge_candidate.centroid_distance),
        "area_ratio": float(edge_candidate.area_ratio),
        "cell_probability_a": float(edge_candidate.cell_probability_a),
        "cell_probability_b": float(edge_candidate.cell_probability_b),
        "row_rank": int(edge_candidate.row_rank),
        "column_rank": int(edge_candidate.column_rank),
        "row_margin": float(edge_candidate.row_margin),
        "column_margin": float(edge_candidate.column_margin),
        "threshold_margin": float(edge_candidate.threshold_margin),
        "activity_similarity": float(edge_candidate.activity_similarity),
        "growth_residual": float(growth.growth_residual),
        "growth_residual_mahalanobis": float(growth.growth_residual_mahalanobis),
        "radial_direction_cosine": float(growth.radial_direction_cosine),
        "expected_area_ratio": float(growth.expected_area_ratio),
        "observed_area_ratio": float(growth.observed_area_ratio),
        "area_growth_residual": float(growth.area_growth_residual),
        "local_neighbor_distortion": float(growth.local_neighbor_distortion),
        "two_edge_motion_consistency": float(growth.two_edge_motion_consistency),
        "two_edge_acceleration": float(growth.two_edge_acceleration),
        "track2p_supported": int(teacher_counts.get(edge, 0) > 0),
        "edge_status_against_gt": (
            "true_positive" if reference_counts.get(edge, 0) > 0 else "false_positive"
        ),
        "pairwise_tp_delta_if_added": int(delta["pairwise_true_positives"]),
        "pairwise_fp_delta_if_added": int(delta["pairwise_false_positives"]),
        "pairwise_fn_delta_if_added": int(delta["pairwise_false_negatives"]),
        "complete_tp_delta_if_added": int(delta["complete_track_true_positives"]),
        "complete_fp_delta_if_added": int(delta["complete_track_false_positives"]),
        "complete_fn_delta_if_added": int(delta["complete_track_false_negatives"]),
        "would_break_complete_tp": int(delta["complete_track_true_positives"] < 0),
        "would_create_complete_fp": int(delta["complete_track_false_positives"] > 0),
        "ranking_score": float(score),
        "is_target_teacher_rescue_edge": int(edge in TARGET_TEACHER_EDGES),
        "candidate_path": _edge_id(edge),
        "path_index": -1,
        "path_step": 1,
        "path_length": 1,
        "path_ranking_score": float(score),
    }


def _assign_ranks(rows: list[dict[str, Any]]) -> None:
    for key, output_name in (
        (lambda row: "ALL", "global_rank"),
        (lambda row: str(row["subject"]), "subject_rank"),
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(key(row))].append(row)
        for group_rows in groups.values():
            group_rows.sort(
                key=lambda row: (
                    -float(row["ranking_score"]),
                    int(row["session_a"]),
                    int(row["session_b"]),
                    int(row["roi_a"]),
                    int(row["roi_b"]),
                )
            )
            for rank_index, row in enumerate(group_rows, start=1):
                row[output_name] = int(rank_index)


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject"])].append(row)
    by_subject["ALL"] = list(rows)
    output: list[dict[str, Any]] = []
    for subject, subject_rows in sorted(by_subject.items()):
        target_rows = [
            row
            for row in subject_rows
            if int(row.get("is_target_teacher_rescue_edge", 0))
        ]
        gt_rows = [
            row
            for row in subject_rows
            if str(row.get("edge_status_against_gt")) == "true_positive"
        ]
        track2p_rows = [
            row for row in subject_rows if int(row.get("track2p_supported", 0)) > 0
        ]
        output.append(
            {
                "subject": subject,
                "candidate_edges": int(len(subject_rows)),
                "track2p_supported_candidates": int(len(track2p_rows)),
                "true_positive_candidates": int(len(gt_rows)),
                "target_teacher_edges_found": int(len(target_rows)),
                "target_teacher_edge_global_ranks": _int_list(
                    tuple(int(row.get("global_rank", -1)) for row in target_rows)
                ),
                "target_teacher_edge_subject_ranks": _int_list(
                    tuple(int(row.get("subject_rank", -1)) for row in target_rows)
                ),
                "target_teacher_edges_top1": int(
                    any(
                        int(row.get("subject_rank", 999999)) <= 1 for row in target_rows
                    )
                ),
                "target_teacher_edges_top3": int(
                    sum(
                        int(row.get("subject_rank", 999999)) <= 3 for row in target_rows
                    )
                ),
                "non_gt_top3_candidates": int(
                    sum(
                        int(row.get("subject_rank", 999999)) <= 3
                        and str(row.get("edge_status_against_gt")) != "true_positive"
                        for row in subject_rows
                    )
                ),
            }
        )
    return output


def _row_observations(predicted: np.ndarray, row_index: int) -> int:
    if row_index < 0 or row_index >= predicted.shape[0]:
        return 0
    return int(np.count_nonzero(np.asarray(predicted[row_index], dtype=int) >= 0))


def _would_complete_predicted_row(
    predicted: np.ndarray, source_row: int, target_row: int, edge: TrackEdge
) -> bool:
    if source_row >= 0 and source_row < predicted.shape[0]:
        row = np.asarray(predicted[source_row], dtype=int).copy()
    elif target_row >= 0 and target_row < predicted.shape[0]:
        row = np.asarray(predicted[target_row], dtype=int).copy()
    else:
        return False
    row[int(edge[0])] = int(edge[2])
    row[int(edge[1])] = int(edge[3])
    if target_row >= 0 and source_row >= 0 and target_row != source_row:
        target = np.asarray(predicted[target_row], dtype=int)
        row = np.maximum(row, target)
    return bool(np.all(row >= 0))


def _finite_or(value: float, default: float) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float(default)


def _nanmin_default(values: Sequence[float], default: float) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(min(finite)) if finite else float(default)


def _int_list(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _edge_list(edges: Sequence[TrackEdge]) -> str:
    return ";".join(_edge_id(edge) for edge in edges)


def _edge_id(edge: TrackEdge) -> str:
    return f"{int(edge[0])}:{int(edge[2])}->{int(edge[1])}:{int(edge[3])}"


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write rows as CSV or JSON."""

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
    """Build the teacher-free adjacent-rescue ranking audit parser."""

    parser = cleanup.build_arg_parser()
    parser.prog = (
        "bayescatrack benchmark "
        "track2p-policy-teacher-free-adjacent-rescue-ranking-audit"
    )
    parser.description = (
        "Rank conflict-free adjacent continuation candidates after the "
        "non-teacher coherence-suffix growth-veto row."
    )
    parser.add_argument(
        "--base-prediction",
        choices=("coherence-suffix-growth-veto", "coherence-suffix"),
        default="coherence-suffix-growth-veto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the teacher-free adjacent-rescue ranking audit CLI."""

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
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        allow_track2p_as_reference_for_smoke_test=(
            args.allow_track2p_as_reference_for_smoke_test
        ),
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    suffix_gate = suffix.CoherenceSuffixStitchGate(
        suffix_path_length=int(args.suffix_path_length),
        min_cell_probability=float(args.min_cell_probability),
        min_area_ratio=float(args.min_area_ratio),
        max_centroid_distance=float(args.max_centroid_distance),
        min_shifted_iou=float(args.min_shifted_iou),
        min_motion_consistency=float(args.min_motion_consistency),
        min_shape_consistency=float(args.min_shape_consistency),
        max_stitches_per_subject=int(args.max_stitches_per_subject),
    )
    growth_gate = cleanup.GrowthVetoGate(
        min_growth_residual_mahalanobis=float(args.min_growth_residual_mahalanobis),
        min_growth_residual=float(args.min_growth_residual),
        min_registered_iou=float(args.min_veto_registered_iou),
        max_registered_iou=args.max_veto_registered_iou,
        min_shifted_iou=float(args.min_veto_shifted_iou),
        max_shifted_iou=args.max_veto_shifted_iou,
        min_cell_probability=float(args.min_veto_cell_probability),
        max_min_cell_probability=args.max_veto_min_cell_probability,
        max_local_neighbor_distortion=args.max_veto_local_neighbor_distortion,
        min_anchor_count=int(args.min_veto_anchor_count),
        min_complete_component_size=args.min_veto_complete_component_size,
        max_row_rank=int(args.max_veto_row_rank),
        max_column_rank=int(args.max_veto_column_rank),
        require_not_suffix_edge=bool(args.require_veto_not_suffix_edge),
        require_terminal_edge=bool(args.require_veto_terminal_edge),
        require_last_session_edge=bool(args.require_veto_last_session_edge),
        require_complete_component=bool(args.require_veto_complete_component),
        max_vetoes_per_subject=int(args.max_vetoes_per_subject),
    )
    result = run_track2p_policy_teacher_free_adjacent_rescue_ranking_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
        suffix_gate=suffix_gate,
        growth_veto_gate=growth_gate,
        base_prediction=cast(BasePrediction, args.base_prediction),
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
        anchor_min_registered_iou=float(args.anchor_min_registered_iou),
        anchor_min_shifted_iou=float(args.anchor_min_shifted_iou),
        anchor_min_cell_probability=float(args.anchor_min_cell_probability),
        progress=bool(args.progress),
    )
    write_rows(
        result.rows,
        args.output,
        output_format=cast(Literal["csv", "json"], args.format),
    )
    if args.summary_output is not None:
        write_rows(
            result.summary_rows,
            args.summary_output,
            output_format=cast(Literal["csv", "json"], args.format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
