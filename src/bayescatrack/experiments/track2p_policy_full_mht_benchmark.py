"""PyRecEst-backed full scan-assignment MHT prototype for Track2p-style data.

This runner is intentionally a prototype, not a new default method.  Unlike the
existing residual-MHT cleanup rows, it opens a *full assignment beam* across
consecutive sessions.  Each global hypothesis is a complete seed-anchored track
table; at every session transition, PyRecEst's Murty ranked assignment utility
generates k-best one-to-one assignments between currently active tracks and the
next-session detections.  The beam is then pruned by accumulated label-free
association cost.

The implementation is deliberately conservative:

* no manual-GT labels enter candidate scoring or hypothesis selection;
* the default benchmark seed set is the reference seed ROI set, matching the
  official seed-restricted scoring protocol.  Use ``--seed-source all-cells`` to
  start tracks from all seed-session cells instead;
* only consecutive transitions are used in this first full-MHT prototype;
* missed detections are allowed through a row non-assignment cost, but tracks
  that miss a scan are not reactivated later in this first version.

This is meant to answer whether a true scan-level MHT formulation is promising
enough to warrant a more complete dynamic model with births/deaths, gap edges,
and growth-aware prediction.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.experiments import track2p_policy_suffix_stitch_ranking_audit as rank
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_growth_field_residual_audit import (
    _cell_probability,
)
from bayescatrack.experiments.track2p_fov_affine_benchmark import (
    _pairwise_iou_matrix_sparse,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    _area_ratio_matrix,
    _mask_centroids,
    _roi_indices,
)

try:
    from pyrecest.utils import murty_k_best_assignments
except ImportError as exc:  # pragma: no cover - stale PyRecEst environment
    raise ImportError(
        "track2p-policy-full-mht requires PyRecEst with "
        "pyrecest.utils.murty_k_best_assignments."
    ) from exc


METHOD = "track2p-policy-full-mht"
SeedSource = Literal["reference", "all-cells"]


@dataclass(frozen=True)
class FullMHTConfig:
    """Controls for the bounded full scan-assignment MHT prototype."""

    beam_width: int = 8
    scan_hypotheses: int = 8
    edge_top_k: int = 4
    miss_cost: float = 2.0
    min_edge_score: float = 0.25
    seed_source: SeedSource = "reference"
    max_seed_tracks: int | None = None
    registered_iou_weight: float = 1.0
    shifted_iou_weight: float = 1.5
    area_ratio_weight: float = 0.25
    cell_probability_weight: float = 0.25
    centroid_distance_weight: float = 0.05
    threshold_margin_weight: float = 0.50


@dataclass(frozen=True)
class _MHTHypothesis:
    """One full-track-table hypothesis in the MHT beam."""

    tracks: np.ndarray
    score: float
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FullMHTResult:
    """Benchmark rows plus diagnostic scan/hypothesis rows."""

    results: tuple[SubjectBenchmarkResult, ...]
    diagnostic_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


def run_track2p_policy_full_mht(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    mht_config: FullMHTConfig | None = None,
    progress: bool = False,
) -> FullMHTResult:
    """Run a bounded full scan-assignment MHT benchmark."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    mht_config = mht_config or FullMHTConfig()
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    diagnostic_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for subject_index, subject_dir in enumerate(subject_dirs, start=1):
        if progress:
            print(
                f"{METHOD}: subject {subject_index}/{len(subject_dirs)} {subject_dir.name}",
                flush=True,
            )
        output = _run_subject_full_mht(
            subject_dir,
            config=policy_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            mht_config=mht_config,
            progress=progress,
        )
        results.append(output["result"])
        diagnostic_rows.extend(output["diagnostic_rows"])
        summary_rows.append(output["summary_row"])
    summary_rows.append(_all_summary_row(summary_rows))
    return FullMHTResult(tuple(results), tuple(diagnostic_rows), tuple(summary_rows))


def _run_subject_full_mht(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    mht_config: FullMHTConfig,
    progress: bool = False,
) -> dict[str, Any]:
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError(f"{METHOD} requires independent manual-GT references")
    sessions = _load_subject_sessions(subject_dir, config)
    _validate_reference_roi_indices(reference, sessions)
    reference_tracks = _reference_matrix(reference, curated_only=config.curated_only)
    n_sessions = len(sessions)
    if n_sessions < 2:
        raise ValueError(f"{subject_dir.name} has fewer than two sessions")

    seed_rois = _seed_rois(
        sessions,
        reference_tracks,
        seed_session=int(config.seed_session),
        seed_source=mht_config.seed_source,
        cell_probability_threshold=float(config.cell_probability_threshold),
    )
    if mht_config.max_seed_tracks is not None:
        seed_rois = seed_rois[: max(0, int(mht_config.max_seed_tracks))]
    if not seed_rois:
        raise ValueError(f"{subject_dir.name}: no seed ROIs for full MHT")

    initial = np.full((len(seed_rois), n_sessions), -1, dtype=int)
    initial[:, int(config.seed_session)] = np.asarray(seed_rois, dtype=int)
    hypotheses: list[_MHTHypothesis] = [
        _MHTHypothesis(initial, 0.0, tuple())
    ]

    feature_cache = rank._FeatureCache(
        sessions=sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(config.cell_probability_threshold),
        matrices={},
    )

    diagnostics: list[dict[str, Any]] = []
    start_session = int(config.seed_session)
    for session_index in range(start_session, n_sessions - 1):
        if progress:
            print(
                f"{METHOD}: {subject_dir.name} scan {session_index}->{session_index + 1} "
                f"input_hypotheses={len(hypotheses)}",
                flush=True,
            )
        hypotheses = _advance_scan(
            hypotheses,
            sessions=sessions,
            feature_cache=feature_cache,
            session_index=session_index,
            config=mht_config,
        )
        if progress:
            print(
                f"{METHOD}: {subject_dir.name} scan {session_index}->{session_index + 1} "
                f"output_hypotheses={len(hypotheses)} best_score={hypotheses[0].score:.6g}",
                flush=True,
            )
        for rank_index, hypothesis in enumerate(hypotheses, start=1):
            last = hypothesis.history[-1] if hypothesis.history else {}
            diagnostics.append(
                {
                    "subject": subject_dir.name,
                    "session_index": int(session_index),
                    "hypothesis_rank": int(rank_index),
                    "hypothesis_score": float(hypothesis.score),
                    "scan_assignment_cost": float(last.get("scan_cost", 0.0)),
                    "scan_assigned_edges": int(last.get("assigned_edges", 0)),
                    "scan_missed_tracks": int(last.get("missed_tracks", 0)),
                    "scan_candidates": int(last.get("scan_candidates", 0)),
                    "beam_width": int(mht_config.beam_width),
                    "scan_hypotheses": int(mht_config.scan_hypotheses),
                    "edge_top_k": int(mht_config.edge_top_k),
                    "seed_source": str(mht_config.seed_source),
                    "n_seed_tracks": int(len(seed_rois)),
                }
            )

    best = hypotheses[0]
    scores = _score_prediction_against_reference(best.tracks, reference, config=config)
    scores = {
        **dict(scores),
        "track2p_full_mht_best_score": float(best.score),
        "track2p_full_mht_beam_width": int(mht_config.beam_width),
        "track2p_full_mht_scan_hypotheses": int(mht_config.scan_hypotheses),
        "track2p_full_mht_edge_top_k": int(mht_config.edge_top_k),
        "track2p_full_mht_miss_cost": float(mht_config.miss_cost),
        "track2p_full_mht_seed_source": str(mht_config.seed_source),
        "track2p_full_mht_n_seed_tracks": int(len(seed_rois)),
    }
    result = SubjectBenchmarkResult(
        subject=subject_dir.name,
        variant="PyRecEst full scan-assignment MHT prototype",
        method=cast(Any, METHOD),
        scores=scores,
        n_sessions=n_sessions,
        reference_source=GROUND_TRUTH_REFERENCE_SOURCE,
    )
    summary_row = {
        "subject": subject_dir.name,
        "n_seed_tracks": int(len(seed_rois)),
        "final_hypotheses": int(len(hypotheses)),
        "best_score": float(best.score),
        "pairwise_f1": float(scores.get("pairwise_f1", float("nan"))),
        "complete_track_f1": float(scores.get("complete_track_f1", float("nan"))),
    }
    return {
        "result": result,
        "diagnostic_rows": diagnostics,
        "summary_row": summary_row,
    }


def _seed_rois(
    sessions: Sequence[Any],
    reference_tracks: np.ndarray,
    *,
    seed_session: int,
    seed_source: SeedSource,
    cell_probability_threshold: float,
) -> list[int]:
    if seed_source == "reference":
        rois = sorted(
            {
                int(row[int(seed_session)])
                for row in np.asarray(reference_tracks, dtype=int)
                if int(row[int(seed_session)]) >= 0
            }
        )
        return rois
    if seed_source != "all-cells":
        raise ValueError(f"Unsupported seed_source: {seed_source!r}")
    output: list[int] = []
    for roi in _roi_indices(sessions[int(seed_session)]):
        if (
            _cell_probability(sessions, int(seed_session), int(roi))
            >= float(cell_probability_threshold)
        ):
            output.append(int(roi))
    return sorted(output)


def _registered_pair(
    sessions: Sequence[Any], feature_cache: rank._FeatureCache, *, session_index: int
) -> Any:
    registered_cache = getattr(feature_cache, "_full_mht_registered_pairs", None)
    if registered_cache is None:
        registered_cache = {}
        setattr(feature_cache, "_full_mht_registered_pairs", registered_cache)
    cached = registered_cache.get(int(session_index))
    if cached is not None:
        return cached
    registered = rank.register_plane_pair(
        sessions[int(session_index)].plane_data,
        sessions[int(session_index) + 1].plane_data,
        transform_type=feature_cache.transform_type,
    )
    registered_cache[int(session_index)] = registered
    return registered


def _sparse_cross_iou_diagnostic_matrices(
    reference_masks: np.ndarray,
    moving_masks: np.ndarray,
    *,
    distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (int(reference_masks.shape[0]), int(moving_masks.shape[0]))
    if reference_masks.shape[0] == 0 or moving_masks.shape[0] == 0:
        return (
            np.zeros(shape, dtype=float),
            np.zeros(shape, dtype=float),
            np.ones(shape, dtype=float),
        )
    reference_masks = np.asarray(reference_masks, dtype=bool)
    moving_masks = np.asarray(moving_masks, dtype=bool)
    distances = np.linalg.norm(
        _mask_centroids(reference_masks)[:, None, :]
        - _mask_centroids(moving_masks)[None, :, :],
        axis=2,
    )
    area_ratios = _area_ratio_matrix(reference_masks, moving_masks)
    iou = _pairwise_iou_matrix_sparse(reference_masks, moving_masks)
    iou = np.where(distances <= float(distance_threshold), iou, 0.0)
    return iou, distances, area_ratios


def _sparse_pair_matrices(
    sessions: Sequence[Any],
    feature_cache: rank._FeatureCache,
    *,
    session_index: int,
    source_rois: Sequence[int],
    edge_top_k: int,
) -> rank._PairMatrices:
    source_rois_tuple = tuple(int(roi) for roi in source_rois)
    sparse_cache = getattr(feature_cache, "_full_mht_sparse_matrices", None)
    if sparse_cache is None:
        sparse_cache = {}
        setattr(feature_cache, "_full_mht_sparse_matrices", sparse_cache)
    cache_key = (int(session_index), source_rois_tuple, int(edge_top_k))
    cached = sparse_cache.get(cache_key)
    if cached is not None:
        return cached

    reference_session = sessions[int(session_index)]
    moving_session = sessions[int(session_index) + 1]
    registered = _registered_pair(sessions, feature_cache, session_index=int(session_index))
    all_source_indices = _roi_indices(reference_session)
    source_positions = {int(roi): idx for idx, roi in enumerate(all_source_indices)}
    selected_positions = [
        source_positions[int(roi)]
        for roi in source_rois_tuple
        if int(roi) in source_positions
    ]
    reference_masks = (np.asarray(reference_session.plane_data.roi_masks) > 0)[
        np.asarray(selected_positions, dtype=int)
    ]
    all_target_indices = _roi_indices(moving_session)
    selected_target_positions = [
        idx
        for idx, roi in enumerate(all_target_indices)
        if _cell_probability(sessions, int(session_index) + 1, int(roi))
        >= float(feature_cache.cell_probability_threshold)
    ]
    target_indices = np.asarray(
        [int(all_target_indices[int(idx)]) for idx in selected_target_positions],
        dtype=int,
    )
    moving_masks = (np.asarray(registered.roi_masks) > 0)[
        np.asarray(selected_target_positions, dtype=int)
    ]
    registered_iou_all, distances_all, area_ratios_all = _sparse_cross_iou_diagnostic_matrices(
        reference_masks,
        moving_masks,
        distance_threshold=float(feature_cache.iou_distance_threshold),
    )
    if distances_all.size:
        nearby_columns = np.flatnonzero(
            np.any(
                distances_all <= float(feature_cache.iou_distance_threshold),
                axis=0,
            )
        )
    else:
        nearby_columns = np.asarray([], dtype=int)
    target_indices = target_indices[nearby_columns]
    moving_masks = moving_masks[nearby_columns]
    registered_iou = registered_iou_all[:, nearby_columns]
    distances = distances_all[:, nearby_columns]
    area_ratios = area_ratios_all[:, nearby_columns]
    shifted = np.zeros_like(registered_iou, dtype=float)
    shift_target_budget = max(1, int(edge_top_k)) * 4
    for row_index in range(reference_masks.shape[0]):
        row_candidates = np.flatnonzero(
            distances[int(row_index)] <= float(feature_cache.iou_distance_threshold)
        )
        if row_candidates.size == 0:
            continue
        row_order = np.lexsort(
            (
                distances[int(row_index), row_candidates],
                -registered_iou[int(row_index), row_candidates],
            )
        )
        selected_columns = row_candidates[row_order[:shift_target_budget]]
        shifted[int(row_index), selected_columns] = rank._pairwise_shifted_iou_from_support(
            reference_masks[int(row_index) : int(row_index) + 1],
            moving_masks[selected_columns],
            radius=2,
        )["shifted_iou"][0]
    matrices = rank._PairMatrices(
        source_indices=np.asarray(source_rois_tuple, dtype=int),
        target_indices=target_indices,
        registered_iou=registered_iou,
        shifted_iou=shifted,
        centroid_distance=distances,
        area_ratio=area_ratios,
        activity_similarity=np.full_like(registered_iou, float("nan"), dtype=float),
        threshold=float(
            rank._assignment_threshold(
                registered_iou, threshold_method=feature_cache.threshold_method
            )
        ),
    )
    sparse_cache[cache_key] = matrices
    return matrices


def _advance_scan(
    hypotheses: Sequence[_MHTHypothesis],
    *,
    sessions: Sequence[Any],
    feature_cache: rank._FeatureCache,
    session_index: int,
    config: FullMHTConfig,
) -> list[_MHTHypothesis]:
    expanded: list[_MHTHypothesis] = []
    for hypothesis in hypotheses:
        expanded.extend(
            _expand_hypothesis_scan(
                hypothesis,
                sessions=sessions,
                feature_cache=feature_cache,
                session_index=int(session_index),
                config=config,
            )
        )
    expanded.sort(key=lambda hyp: -float(hyp.score))
    return expanded[: max(1, int(config.beam_width))]


def _expand_hypothesis_scan(
    hypothesis: _MHTHypothesis,
    *,
    sessions: Sequence[Any],
    feature_cache: rank._FeatureCache,
    session_index: int,
    config: FullMHTConfig,
) -> list[_MHTHypothesis]:
    tracks = np.asarray(hypothesis.tracks, dtype=int)
    next_session = int(session_index) + 1
    active_rows = [
        row_index
        for row_index, row in enumerate(tracks)
        if int(row[int(session_index)]) >= 0
    ]
    if not active_rows:
        carried = tracks.copy()
        return [
            _MHTHypothesis(
                carried,
                hypothesis.score,
                hypothesis.history
                + (
                    {
                        "session_index": int(session_index),
                        "scan_cost": 0.0,
                        "assigned_edges": 0,
                        "missed_tracks": 0,
                        "scan_candidates": 0,
                    },
                ),
            )
        ]

    matrices = _sparse_pair_matrices(
        sessions,
        feature_cache,
        session_index=int(session_index),
        source_rois=[int(tracks[int(row_index), int(session_index)]) for row_index in active_rows],
        edge_top_k=int(config.edge_top_k),
    )
    source_lookup = {int(roi): idx for idx, roi in enumerate(matrices.source_indices)}
    target_indices = np.asarray(matrices.target_indices, dtype=int)
    finite_targets = [
        int(col)
        for col, target_roi in enumerate(target_indices)
        if _cell_probability(sessions, next_session, int(target_roi))
        >= float(feature_cache.cell_probability_threshold)
    ]
    if not finite_targets:
        carried = tracks.copy()
        carried[active_rows, next_session] = -1
        return [
            _MHTHypothesis(
                carried,
                hypothesis.score - float(config.miss_cost) * float(len(active_rows)),
                hypothesis.history
                + (
                    {
                        "session_index": int(session_index),
                        "scan_cost": float(config.miss_cost) * float(len(active_rows)),
                        "assigned_edges": 0,
                        "missed_tracks": int(len(active_rows)),
                        "scan_candidates": 0,
                    },
                ),
            )
        ]

    cost_matrix = np.full((len(active_rows), len(finite_targets)), np.inf, dtype=float)
    candidate_count = 0
    for row_pos, row_index in enumerate(active_rows):
        source_roi = int(tracks[int(row_index), int(session_index)])
        source_local = source_lookup.get(source_roi)
        if source_local is None:
            continue
        row_scores: list[tuple[float, int]] = []
        for compact_col, target_local in enumerate(finite_targets):
            score = _edge_score(
                sessions,
                matrices,
                session_index=int(session_index),
                source_local=int(source_local),
                target_local=int(target_local),
                config=config,
            )
            if score >= float(config.min_edge_score):
                row_scores.append((float(score), int(compact_col)))
        row_scores.sort(reverse=True)
        for score, compact_col in row_scores[: max(1, int(config.edge_top_k))]:
            cost_matrix[row_pos, compact_col] = -float(score)
            candidate_count += 1

    if not np.isfinite(cost_matrix).any():
        carried = tracks.copy()
        carried[active_rows, next_session] = -1
        return [
            _MHTHypothesis(
                carried,
                hypothesis.score - float(config.miss_cost) * float(len(active_rows)),
                hypothesis.history
                + (
                    {
                        "session_index": int(session_index),
                        "scan_cost": float(config.miss_cost) * float(len(active_rows)),
                        "assigned_edges": 0,
                        "missed_tracks": int(len(active_rows)),
                        "scan_candidates": int(candidate_count),
                    },
                ),
            )
        ]

    solutions = murty_k_best_assignments(
        cost_matrix,
        k=max(1, int(config.scan_hypotheses)),
        row_non_assignment_costs=np.full((len(active_rows),), float(config.miss_cost)),
        col_non_assignment_costs=np.zeros((len(finite_targets),), dtype=float),
    )
    output: list[_MHTHypothesis] = []
    for solution in solutions:
        assignment = np.asarray(solution["assignment"], dtype=int)
        updated = tracks.copy()
        assigned_edges = 0
        missed_tracks = 0
        for row_pos, row_index in enumerate(active_rows):
            compact_col = int(assignment[int(row_pos)])
            if compact_col >= 0:
                target_local = finite_targets[compact_col]
                updated[int(row_index), next_session] = int(target_indices[target_local])
                assigned_edges += 1
            else:
                updated[int(row_index), next_session] = -1
                missed_tracks += 1
        scan_cost = float(solution["cost"])
        output.append(
            _MHTHypothesis(
                updated,
                float(hypothesis.score) - scan_cost,
                hypothesis.history
                + (
                    {
                        "session_index": int(session_index),
                        "scan_cost": scan_cost,
                        "assigned_edges": int(assigned_edges),
                        "missed_tracks": int(missed_tracks),
                        "scan_candidates": int(candidate_count),
                    },
                ),
            )
        )
    return output


def _edge_score(
    sessions: Sequence[Any],
    matrices: rank._PairMatrices,
    *,
    session_index: int,
    source_local: int,
    target_local: int,
    config: FullMHTConfig,
) -> float:
    registered = _finite_float(matrices.registered_iou[source_local, target_local], 0.0)
    shifted = _finite_float(matrices.shifted_iou[source_local, target_local], 0.0)
    centroid = _finite_float(matrices.centroid_distance[source_local, target_local], 1e3)
    area_ratio = _finite_float(matrices.area_ratio[source_local, target_local], 0.0)
    target_roi = int(matrices.target_indices[int(target_local)])
    cell_b = _cell_probability(sessions, int(session_index) + 1, target_roi)
    threshold_margin = registered - float(matrices.threshold)
    score = 0.0
    score += float(config.registered_iou_weight) * registered
    score += float(config.shifted_iou_weight) * shifted
    score += float(config.area_ratio_weight) * max(0.0, min(1.0, area_ratio))
    score += float(config.cell_probability_weight) * max(0.0, min(1.0, cell_b))
    score += float(config.threshold_margin_weight) * max(0.0, threshold_margin)
    score -= float(config.centroid_distance_weight) * max(0.0, centroid)
    return float(score)


def _all_summary_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"subject": "ALL"}
    return {
        "subject": "ALL",
        "n_seed_tracks": int(sum(int(row.get("n_seed_tracks", 0)) for row in rows)),
        "final_hypotheses": int(sum(int(row.get("final_hypotheses", 0)) for row in rows)),
        "best_score": float(sum(float(row.get("best_score", 0.0)) for row in rows)),
    }


def _finite_float(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return numeric if np.isfinite(numeric) else float(fallback)


def _write_rows(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    output_format: Literal["csv", "json"],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
        return
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    import csv

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the full-MHT benchmark parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-full-mht",
        description="Run a PyRecEst Murty scan-assignment full-MHT prototype.",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        default="manual-gt",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument("--input-format", choices=("auto", "suite2p", "npy"), default="suite2p")
    parser.add_argument("--threshold-method", choices=("otsu", "min"), default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD)
    parser.add_argument("--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE)
    parser.add_argument("--iou-distance-threshold", type=float, default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD)
    parser.add_argument("--cell-probability-threshold", type=float, default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD)
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument("--restrict-to-reference-seed-rois", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-track2p-as-reference-for-smoke-test", action="store_true")
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--scan-hypotheses", type=int, default=8)
    parser.add_argument("--edge-top-k", type=int, default=4)
    parser.add_argument("--miss-cost", type=float, default=2.0)
    parser.add_argument("--min-edge-score", type=float, default=0.25)
    parser.add_argument("--seed-source", choices=("reference", "all-cells"), default="reference")
    parser.add_argument("--max-seed-tracks", type=int, default=None)
    parser.add_argument("--registered-iou-weight", type=float, default=1.0)
    parser.add_argument("--shifted-iou-weight", type=float, default=1.5)
    parser.add_argument("--area-ratio-weight", type=float, default=0.25)
    parser.add_argument("--cell-probability-weight", type=float, default=0.25)
    parser.add_argument("--centroid-distance-weight", type=float, default=0.05)
    parser.add_argument("--threshold-margin-weight", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="csv")
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--diagnostics-format", choices=("csv", "json"), default="csv")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the full-MHT benchmark CLI."""

    args = build_arg_parser().parse_args(argv)
    benchmark_config = Track2pBenchmarkConfig(
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
    result = run_track2p_policy_full_mht(
        benchmark_config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        mht_config=FullMHTConfig(
            beam_width=max(1, int(args.beam_width)),
            scan_hypotheses=max(1, int(args.scan_hypotheses)),
            edge_top_k=max(1, int(args.edge_top_k)),
            miss_cost=float(args.miss_cost),
            min_edge_score=float(args.min_edge_score),
            seed_source=cast(SeedSource, args.seed_source),
            max_seed_tracks=args.max_seed_tracks,
            registered_iou_weight=float(args.registered_iou_weight),
            shifted_iou_weight=float(args.shifted_iou_weight),
            area_ratio_weight=float(args.area_ratio_weight),
            cell_probability_weight=float(args.cell_probability_weight),
            centroid_distance_weight=float(args.centroid_distance_weight),
            threshold_margin_weight=float(args.threshold_margin_weight),
        ),
        progress=bool(args.progress),
    )
    write_results(
        [benchmark_result.to_dict() for benchmark_result in result.results],
        args.output,
        cast(OutputFormat, args.format),
    )
    if args.diagnostics_output is not None:
        _write_rows(
            result.diagnostic_rows,
            args.diagnostics_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    if args.summary_output is not None:
        _write_rows(
            result.summary_rows,
            args.summary_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
