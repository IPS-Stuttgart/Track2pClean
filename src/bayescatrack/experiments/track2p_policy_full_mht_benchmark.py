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
* consecutive transitions are used, with bounded one-scan gap reactivation;
* missed detections are allowed through a row non-assignment cost, and low-support
  output histories can be pruned as dead tracks before scoring.

This is meant to answer whether a true scan-level MHT formulation is promising
enough to warrant a more complete dynamic model with births/deaths, gap edges,
and growth-aware prediction.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.growth_priors import fit_affine_growth_transform
from bayescatrack.experiments import track2p_policy_suffix_stitch_ranking_audit as rank
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _predict_subject_tracks,
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
SeedSource = Literal["reference", "all-cells", "track2p-output"]


@dataclass(frozen=True)
class FullMHTConfig:
    """Controls for the bounded full scan-assignment MHT prototype."""

    beam_width: int = 8
    scan_hypotheses: int = 8
    edge_top_k: int = 4
    miss_cost: float = 2.0
    max_gap: int = 1
    gap_reactivation_cost: float = 1.0
    min_output_observations: int = 1
    min_edge_score: float = 0.25
    seed_source: SeedSource = "reference"
    max_seed_tracks: int | None = None
    registered_iou_weight: float = 1.0
    shifted_iou_weight: float = 1.5
    area_ratio_weight: float = 0.25
    cell_probability_weight: float = 0.25
    centroid_distance_weight: float = 0.05
    threshold_margin_weight: float = 0.50
    growth_residual_weight: float = 0.10
    growth_mahalanobis_weight: float = 0.25
    local_deformation_weight: float = 0.50
    track2p_prior_weight: float = 0.0
    track2p_non_prior_penalty: float = 0.0
    track2p_prior_miss_penalty: float = 0.0
    growth_anchor_min_registered_iou: float = 0.55
    growth_anchor_min_shifted_iou: float = 0.30
    growth_anchor_min_cell_probability: float = 0.80


@dataclass(frozen=True)
class _MHTHypothesis:
    """One full-track-table hypothesis in the MHT beam."""

    tracks: np.ndarray
    score: float
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _ActiveTrackSource:
    row_index: int
    source_session: int
    source_roi: int
    gap_length: int


@dataclass(frozen=True)
class FullMHTResult:
    """Benchmark rows plus diagnostic scan/hypothesis rows."""

    results: tuple[SubjectBenchmarkResult, ...]
    diagnostic_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _FullMHTPairMatrices:
    source_session: int
    target_session: int
    source_indices: np.ndarray
    target_indices: np.ndarray
    registered_iou: np.ndarray
    shifted_iou: np.ndarray
    centroid_distance: np.ndarray
    area_ratio: np.ndarray
    threshold: float
    growth_residual: np.ndarray
    growth_mahalanobis: np.ndarray
    local_deformation: np.ndarray
    growth_anchor_count: int
    growth_model_type: str


@dataclass(frozen=True)
class _GrowthPrior:
    affine_xy: np.ndarray
    covariance_inverse: np.ndarray
    anchor_count: int
    model_type: str


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

    track2p_prediction = (
        _track2p_prediction_for_subject(subject_dir, config=config)
        if (
            mht_config.seed_source == "track2p-output"
            or float(mht_config.track2p_prior_weight) != 0.0
        )
        else None
    )
    seed_rois = _seed_rois(
        sessions,
        reference_tracks,
        seed_session=int(config.seed_session),
        seed_source=mht_config.seed_source,
        cell_probability_threshold=float(config.cell_probability_threshold),
        track2p_tracks=track2p_prediction,
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
    track2p_prior_edges = _track2p_prior_edges(
        subject_dir,
        config=config,
        enabled=float(mht_config.track2p_prior_weight) != 0.0,
        track2p_tracks=track2p_prediction,
    )

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
            track2p_prior_edges=track2p_prior_edges,
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
                    "scan_selected_prior_edges": int(
                        last.get("selected_prior_edges", 0)
                    ),
                    "scan_selected_non_prior_edges": int(
                        last.get("selected_non_prior_edges", 0)
                    ),
                    "scan_missed_prior_successors": int(
                        last.get("missed_prior_successors", 0)
                    ),
                    "scan_selected_edge_summaries": str(
                        last.get("selected_edge_summaries", "")
                    ),
                    "scan_gap_active_tracks": int(last.get("gap_active_tracks", 0)),
                    "scan_gap_reactivated_tracks": int(
                        last.get("gap_reactivated_tracks", 0)
                    ),
                    "scan_max_gap_length": int(last.get("max_gap_length", 0)),
                    "scan_candidates": int(last.get("scan_candidates", 0)),
                    "scan_growth_anchor_count": int(
                        last.get("growth_anchor_count", 0)
                    ),
                    "scan_growth_model_type": str(
                        last.get("growth_model_type", "")
                    ),
                    "beam_width": int(mht_config.beam_width),
                    "scan_hypotheses": int(mht_config.scan_hypotheses),
                    "edge_top_k": int(mht_config.edge_top_k),
                    "seed_source": str(mht_config.seed_source),
                    "n_seed_tracks": int(len(seed_rois)),
                }
            )

    best = hypotheses[0]
    output_tracks = _prune_output_tracks(
        best.tracks, min_observations=int(mht_config.min_output_observations)
    )
    scores = _score_prediction_against_reference(output_tracks, reference, config=config)
    scores = {
        **dict(scores),
        "track2p_full_mht_best_score": float(best.score),
        "track2p_full_mht_beam_width": int(mht_config.beam_width),
        "track2p_full_mht_scan_hypotheses": int(mht_config.scan_hypotheses),
        "track2p_full_mht_edge_top_k": int(mht_config.edge_top_k),
        "track2p_full_mht_miss_cost": float(mht_config.miss_cost),
        "track2p_full_mht_max_gap": int(mht_config.max_gap),
        "track2p_full_mht_gap_reactivation_cost": float(
            mht_config.gap_reactivation_cost
        ),
        "track2p_full_mht_min_output_observations": int(
            mht_config.min_output_observations
        ),
        "track2p_full_mht_growth_residual_weight": float(
            mht_config.growth_residual_weight
        ),
        "track2p_full_mht_growth_mahalanobis_weight": float(
            mht_config.growth_mahalanobis_weight
        ),
        "track2p_full_mht_local_deformation_weight": float(
            mht_config.local_deformation_weight
        ),
        "track2p_full_mht_track2p_prior_weight": float(
            mht_config.track2p_prior_weight
        ),
        "track2p_full_mht_track2p_non_prior_penalty": float(
            mht_config.track2p_non_prior_penalty
        ),
        "track2p_full_mht_track2p_prior_miss_penalty": float(
            mht_config.track2p_prior_miss_penalty
        ),
        "track2p_full_mht_growth_anchor_min_registered_iou": float(
            mht_config.growth_anchor_min_registered_iou
        ),
        "track2p_full_mht_growth_anchor_min_shifted_iou": float(
            mht_config.growth_anchor_min_shifted_iou
        ),
        "track2p_full_mht_growth_anchor_min_cell_probability": float(
            mht_config.growth_anchor_min_cell_probability
        ),
        "track2p_full_mht_seed_source": str(mht_config.seed_source),
        "track2p_full_mht_n_seed_tracks": int(len(seed_rois)),
        "track2p_full_mht_n_output_tracks": int(output_tracks.shape[0]),
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
    track2p_tracks: np.ndarray | None = None,
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
    if seed_source == "track2p-output":
        if track2p_tracks is None:
            raise ValueError("track2p-output seed source requires Track2p tracks")
        tracks = np.asarray(track2p_tracks, dtype=int)
        if tracks.ndim != 2 or int(seed_session) >= tracks.shape[1]:
            return []
        return sorted(
            {
                int(row[int(seed_session)])
                for row in tracks
                if int(row[int(seed_session)]) >= 0
                and _cell_probability(
                    sessions, int(seed_session), int(row[int(seed_session)])
                )
                >= float(cell_probability_threshold)
            }
        )
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


def _track2p_prior_edges(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    enabled: bool,
    track2p_tracks: np.ndarray | None = None,
) -> frozenset[tuple[int, int, int, int]]:
    if not enabled:
        return frozenset()
    predicted = (
        np.asarray(track2p_tracks, dtype=int)
        if track2p_tracks is not None
        else _track2p_prediction_for_subject(subject_dir, config=config)
    )
    return _track_edges(np.asarray(predicted, dtype=int))


def _track2p_prediction_for_subject(
    subject_dir: Path, *, config: Track2pBenchmarkConfig
) -> np.ndarray:
    baseline_config = replace(config, method="track2p-baseline")
    predicted, _variant = _predict_subject_tracks(subject_dir, baseline_config)
    return np.asarray(predicted, dtype=int)


def _track_edges(matrix: np.ndarray) -> frozenset[tuple[int, int, int, int]]:
    tracks = np.asarray(matrix, dtype=int)
    edges: set[tuple[int, int, int, int]] = set()
    if tracks.ndim != 2 or tracks.shape[1] < 2:
        return frozenset()
    for row in tracks:
        for session_index in range(tracks.shape[1] - 1):
            roi_a = int(row[int(session_index)])
            roi_b = int(row[int(session_index) + 1])
            if roi_a >= 0 and roi_b >= 0:
                edges.add((int(session_index), int(session_index) + 1, roi_a, roi_b))
    return frozenset(edges)


def _prune_output_tracks(matrix: np.ndarray, *, min_observations: int) -> np.ndarray:
    tracks = np.asarray(matrix, dtype=int)
    if tracks.ndim != 2:
        return tracks.reshape(0, 0)
    threshold = max(1, int(min_observations))
    if threshold <= 1:
        return tracks
    keep = np.sum(tracks >= 0, axis=1) >= threshold
    return tracks[np.asarray(keep, dtype=bool)]


def _proposal_target_rois(
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    *,
    source_session: int,
    target_session: int,
    source_rois: Sequence[int],
) -> tuple[int, ...]:
    if not track2p_prior_edges:
        return tuple()
    source_roi_set = {int(roi) for roi in source_rois}
    return tuple(
        sorted(
            {
                int(roi_b)
                for session_a, session_b, roi_a, roi_b in track2p_prior_edges
                if int(session_a) == int(source_session)
                and int(session_b) == int(target_session)
                and int(roi_a) in source_roi_set
            }
        )
    )


def _registered_pair(
    sessions: Sequence[Any],
    feature_cache: rank._FeatureCache,
    *,
    source_session: int,
    target_session: int,
) -> Any:
    registered_cache = getattr(feature_cache, "_full_mht_registered_pairs", None)
    if registered_cache is None:
        registered_cache = {}
        setattr(feature_cache, "_full_mht_registered_pairs", registered_cache)
    cache_key = (int(source_session), int(target_session))
    cached = registered_cache.get(cache_key)
    if cached is not None:
        return cached
    registered = rank.register_plane_pair(
        sessions[int(source_session)].plane_data,
        sessions[int(target_session)].plane_data,
        transform_type=feature_cache.transform_type,
    )
    registered_cache[cache_key] = registered
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
    source_session: int,
    target_session: int,
    source_rois: Sequence[int],
    edge_top_k: int,
    config: FullMHTConfig,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]] = frozenset(),
) -> _FullMHTPairMatrices:
    source_rois_tuple = tuple(int(roi) for roi in source_rois)
    proposal_target_rois = _proposal_target_rois(
        track2p_prior_edges,
        source_session=int(source_session),
        target_session=int(target_session),
        source_rois=source_rois_tuple,
    )
    sparse_cache = getattr(feature_cache, "_full_mht_sparse_matrices", None)
    if sparse_cache is None:
        sparse_cache = {}
        setattr(feature_cache, "_full_mht_sparse_matrices", sparse_cache)
    cache_key = (
        int(source_session),
        int(target_session),
        source_rois_tuple,
        proposal_target_rois,
        int(edge_top_k),
        float(config.growth_anchor_min_registered_iou),
        float(config.growth_anchor_min_shifted_iou),
        float(config.growth_anchor_min_cell_probability),
    )
    cached = sparse_cache.get(cache_key)
    if cached is not None:
        return cached

    reference_session = sessions[int(source_session)]
    moving_session = sessions[int(target_session)]
    registered = _registered_pair(
        sessions,
        feature_cache,
        source_session=int(source_session),
        target_session=int(target_session),
    )
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
        if (
            _cell_probability(sessions, int(target_session), int(roi))
            >= float(feature_cache.cell_probability_threshold)
            or int(roi) in proposal_target_rois
        )
    ]
    target_indices = np.asarray(
        [int(all_target_indices[int(idx)]) for idx in selected_target_positions],
        dtype=int,
    )
    target_cell_probabilities = np.asarray(
        [
            _cell_probability(sessions, int(target_session), int(roi))
            for roi in target_indices
        ],
        dtype=float,
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
        if proposal_target_rois:
            proposal_columns = np.flatnonzero(
                np.isin(target_indices, np.asarray(proposal_target_rois, dtype=int))
            )
            nearby_columns = np.union1d(nearby_columns, proposal_columns)
    else:
        nearby_columns = np.asarray([], dtype=int)
    target_indices = target_indices[nearby_columns]
    target_cell_probabilities = target_cell_probabilities[nearby_columns]
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
    source_centroids = _mask_centroids(reference_masks)
    target_centroids = _mask_centroids(moving_masks)
    (
        growth_residual,
        growth_mahalanobis,
        local_deformation,
        growth_prior,
    ) = _growth_context_matrices(
        source_centroids,
        target_centroids,
        registered_iou=registered_iou,
        shifted_iou=shifted,
        target_cell_probabilities=target_cell_probabilities,
        config=config,
    )
    matrices = _FullMHTPairMatrices(
        source_session=int(source_session),
        target_session=int(target_session),
        source_indices=np.asarray(source_rois_tuple, dtype=int),
        target_indices=target_indices,
        registered_iou=registered_iou,
        shifted_iou=shifted,
        centroid_distance=distances,
        area_ratio=area_ratios,
        threshold=float(
            rank._assignment_threshold(
                registered_iou, threshold_method=feature_cache.threshold_method
            )
        ),
        growth_residual=growth_residual,
        growth_mahalanobis=growth_mahalanobis,
        local_deformation=local_deformation,
        growth_anchor_count=int(growth_prior.anchor_count),
        growth_model_type=str(growth_prior.model_type),
    )
    sparse_cache[cache_key] = matrices
    return matrices


def _growth_context_matrices(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    *,
    registered_iou: np.ndarray,
    shifted_iou: np.ndarray,
    target_cell_probabilities: np.ndarray,
    config: FullMHTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _GrowthPrior]:
    shape = np.asarray(registered_iou, dtype=float).shape
    if len(shape) != 2:
        raise ValueError("registered_iou must be a 2-D matrix")
    source_xy = np.asarray(source_centroids, dtype=float).reshape(shape[0], 2)
    target_xy = np.asarray(target_centroids, dtype=float).reshape(shape[1], 2)
    anchor_pairs = _mutual_growth_anchor_pairs(
        registered_iou=np.asarray(registered_iou, dtype=float),
        shifted_iou=np.asarray(shifted_iou, dtype=float),
        target_cell_probabilities=np.asarray(target_cell_probabilities, dtype=float),
        config=config,
    )
    prior = _fit_growth_prior_from_anchor_pairs(source_xy, target_xy, anchor_pairs)
    if shape[0] == 0 or shape[1] == 0:
        empty = np.zeros(shape, dtype=float)
        return empty, empty, empty, prior
    predicted = _apply_affine_points(source_xy, prior.affine_xy)
    residual_vectors = target_xy[None, :, :] - predicted[:, None, :]
    residual = np.linalg.norm(residual_vectors, axis=2)
    mahalanobis_squared = np.einsum(
        "...i,ij,...j->...",
        residual_vectors,
        prior.covariance_inverse,
        residual_vectors,
    )
    mahalanobis = np.sqrt(np.maximum(0.0, mahalanobis_squared))
    local_deformation = _local_deformation_matrix(
        source_xy,
        target_xy,
        anchor_pairs=anchor_pairs,
        affine_xy=prior.affine_xy,
    )
    return (
        residual.astype(float),
        mahalanobis.astype(float),
        local_deformation.astype(float),
        prior,
    )


def _growth_residual_matrices(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    *,
    registered_iou: np.ndarray,
    shifted_iou: np.ndarray,
    target_cell_probabilities: np.ndarray,
    config: FullMHTConfig,
) -> tuple[np.ndarray, np.ndarray, _GrowthPrior]:
    residual, mahalanobis, _local, prior = _growth_context_matrices(
        source_centroids,
        target_centroids,
        registered_iou=registered_iou,
        shifted_iou=shifted_iou,
        target_cell_probabilities=target_cell_probabilities,
        config=config,
    )
    return residual, mahalanobis, prior


def _fit_growth_prior_from_candidate_matrices(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    *,
    registered_iou: np.ndarray,
    shifted_iou: np.ndarray,
    target_cell_probabilities: np.ndarray,
    config: FullMHTConfig,
) -> _GrowthPrior:
    anchor_pairs = _mutual_growth_anchor_pairs(
        registered_iou=registered_iou,
        shifted_iou=shifted_iou,
        target_cell_probabilities=target_cell_probabilities,
        config=config,
    )
    return _fit_growth_prior_from_anchor_pairs(
        np.asarray(source_centroids, dtype=float),
        np.asarray(target_centroids, dtype=float),
        anchor_pairs,
    )


def _fit_growth_prior_from_anchor_pairs(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    anchor_pairs: Sequence[tuple[int, int]],
) -> _GrowthPrior:
    if not anchor_pairs:
        return _identity_growth_prior()
    anchor_rows = np.asarray([row for row, _col in anchor_pairs], dtype=int)
    anchor_cols = np.asarray([col for _row, col in anchor_pairs], dtype=int)
    return _fit_growth_prior_from_points(
        np.asarray(source_centroids, dtype=float)[anchor_rows],
        np.asarray(target_centroids, dtype=float)[anchor_cols],
    )


def _mutual_growth_anchor_pairs(
    *,
    registered_iou: np.ndarray,
    shifted_iou: np.ndarray,
    target_cell_probabilities: np.ndarray,
    config: FullMHTConfig,
) -> tuple[tuple[int, int], ...]:
    registered = np.asarray(registered_iou, dtype=float)
    shifted = np.asarray(shifted_iou, dtype=float)
    if registered.size == 0 or registered.shape != shifted.shape:
        return tuple()
    cell_prob = np.asarray(target_cell_probabilities, dtype=float).reshape(-1)
    if cell_prob.shape[0] != registered.shape[1]:
        return tuple()
    valid = (
        np.isfinite(registered)
        & np.isfinite(shifted)
        & (registered >= float(config.growth_anchor_min_registered_iou))
        & (shifted >= float(config.growth_anchor_min_shifted_iou))
        & (cell_prob[None, :] >= float(config.growth_anchor_min_cell_probability))
    )
    scores = np.where(valid, registered + shifted, -np.inf)
    if not np.isfinite(scores).any():
        return tuple()
    row_best = np.argmax(scores, axis=1)
    col_best = np.argmax(scores, axis=0)
    pairs: list[tuple[int, int]] = []
    for row, col in enumerate(row_best):
        if not np.isfinite(scores[int(row), int(col)]):
            continue
        if int(col_best[int(col)]) == int(row):
            pairs.append((int(row), int(col)))
    return tuple(pairs)


def _fit_growth_prior_from_points(
    source_xy: np.ndarray, target_xy: np.ndarray
) -> _GrowthPrior:
    source_xy = np.asarray(source_xy, dtype=float).reshape(-1, 2)
    target_xy = np.asarray(target_xy, dtype=float).reshape(-1, 2)
    if source_xy.shape[0] == 0 or target_xy.shape[0] == 0:
        return _identity_growth_prior()
    if source_xy.shape[0] >= 3:
        affine = fit_affine_growth_transform(source_xy, target_xy)
        residual_vectors = target_xy - _apply_affine_points(source_xy, affine)
        residual_norms = np.linalg.norm(residual_vectors, axis=1)
        median = float(np.median(residual_norms))
        mad = float(np.median(np.abs(residual_norms - median)))
        keep = residual_norms <= median + max(3.0 * 1.4826 * mad, 2.0)
        if int(np.sum(keep)) >= 3 and not bool(np.all(keep)):
            affine = fit_affine_growth_transform(source_xy[keep], target_xy[keep])
            residual_vectors = target_xy[keep] - _apply_affine_points(
                source_xy[keep], affine
            )
            model_type = "robust_affine"
        else:
            model_type = "affine"
    else:
        displacement = np.median(target_xy - source_xy, axis=0)
        affine = np.asarray(
            [[1.0, 0.0, float(displacement[0])], [0.0, 1.0, float(displacement[1])]],
            dtype=float,
        )
        residual_vectors = target_xy - _apply_affine_points(source_xy, affine)
        model_type = "translation_fallback"
    return _GrowthPrior(
        affine_xy=np.asarray(affine, dtype=float).reshape(2, 3),
        covariance_inverse=_growth_covariance_inverse(residual_vectors),
        anchor_count=int(source_xy.shape[0]),
        model_type=model_type,
    )


def _identity_growth_prior() -> _GrowthPrior:
    return _GrowthPrior(
        affine_xy=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        covariance_inverse=np.eye(2, dtype=float),
        anchor_count=0,
        model_type="identity_no_anchors",
    )


def _apply_affine_points(points_xy: np.ndarray, affine_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    affine = np.asarray(affine_xy, dtype=float).reshape(2, 3)
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=float)])
    return homogeneous @ affine.T


def _local_deformation_matrix(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    *,
    anchor_pairs: Sequence[tuple[int, int]],
    affine_xy: np.ndarray,
) -> np.ndarray:
    source_xy = np.asarray(source_centroids, dtype=float).reshape(-1, 2)
    target_xy = np.asarray(target_centroids, dtype=float).reshape(-1, 2)
    output = np.zeros((source_xy.shape[0], target_xy.shape[0]), dtype=float)
    if not anchor_pairs or source_xy.shape[0] == 0 or target_xy.shape[0] == 0:
        return output
    linear = np.asarray(affine_xy, dtype=float).reshape(2, 3)[:, :2]
    anchor_rows = np.asarray([row for row, _col in anchor_pairs], dtype=int)
    anchor_cols = np.asarray([col for _row, col in anchor_pairs], dtype=int)
    anchor_source = source_xy[anchor_rows]
    anchor_target = target_xy[anchor_cols]
    for source_index, source in enumerate(source_xy):
        source_deltas = source[None, :] - anchor_source
        predicted_deltas = source_deltas @ linear.T
        predicted_norms = np.linalg.norm(predicted_deltas, axis=1)
        usable = predicted_norms > 1.0e-9
        if not np.any(usable):
            continue
        observed_deltas = target_xy[:, None, :] - anchor_target[None, usable, :]
        residuals = np.linalg.norm(
            observed_deltas - predicted_deltas[None, usable, :], axis=2
        ) / np.maximum(predicted_norms[None, usable], 1.0)
        output[int(source_index), :] = np.median(residuals, axis=1)
    return output


def _growth_covariance_inverse(residual_vectors: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residual_vectors, dtype=float).reshape(-1, 2)
    finite = np.all(np.isfinite(residuals), axis=1)
    residuals = residuals[finite]
    residual_scale = _growth_residual_scale(residuals)
    if residuals.shape[0] >= 2:
        covariance = np.cov(residuals.T)
    else:
        covariance = np.eye(2, dtype=float) * residual_scale**2
    covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
    covariance += np.eye(2, dtype=float) * max(1.0e-6, residual_scale**2)
    return np.linalg.pinv(covariance)


def _growth_residual_scale(residual_vectors: np.ndarray) -> float:
    residuals = np.asarray(residual_vectors, dtype=float).reshape(-1, 2)
    if residuals.size == 0:
        return 1.0
    norms = np.linalg.norm(residuals, axis=1)
    norms = norms[np.isfinite(norms)]
    if norms.size == 0:
        return 1.0
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    return max(1.0, median + 1.4826 * mad)


def _active_track_sources(
    tracks: np.ndarray, *, session_index: int, max_gap: int
) -> tuple[_ActiveTrackSource, ...]:
    matrix = np.asarray(tracks, dtype=int)
    current = int(session_index)
    horizon = max(0, int(max_gap))
    output: list[_ActiveTrackSource] = []
    for row_index, row in enumerate(matrix):
        observed = np.flatnonzero(np.asarray(row[: current + 1], dtype=int) >= 0)
        if observed.size == 0:
            continue
        source_session = int(observed[-1])
        gap_length = int(current - source_session)
        if gap_length > horizon:
            continue
        output.append(
            _ActiveTrackSource(
                row_index=int(row_index),
                source_session=source_session,
                source_roi=int(row[source_session]),
                gap_length=gap_length,
            )
        )
    return tuple(output)


def _matrix_diagnostics(
    matrices_by_source_session: Mapping[int, _FullMHTPairMatrices],
) -> dict[str, Any]:
    if not matrices_by_source_session:
        return {"growth_anchor_count": 0, "growth_model_type": "no_active_tracks"}
    anchor_count = sum(
        int(matrices.growth_anchor_count)
        for matrices in matrices_by_source_session.values()
    )
    model_types = ";".join(
        f"{int(source_session)}:{matrices.growth_model_type}"
        for source_session, matrices in sorted(matrices_by_source_session.items())
    )
    return {
        "growth_anchor_count": int(anchor_count),
        "growth_model_type": model_types,
    }


def _advance_scan(
    hypotheses: Sequence[_MHTHypothesis],
    *,
    sessions: Sequence[Any],
    feature_cache: rank._FeatureCache,
    session_index: int,
    config: FullMHTConfig,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
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
                track2p_prior_edges=track2p_prior_edges,
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
    track2p_prior_edges: frozenset[tuple[int, int, int, int]] | None = None,
) -> list[_MHTHypothesis]:
    tracks = np.asarray(hypothesis.tracks, dtype=int)
    next_session = int(session_index) + 1
    prior_edges = track2p_prior_edges or frozenset()
    active_sources = _active_track_sources(
        tracks, session_index=int(session_index), max_gap=int(config.max_gap)
    )
    if not active_sources:
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
                        "gap_active_tracks": 0,
                        "gap_reactivated_tracks": 0,
                        "max_gap_length": 0,
                        "scan_candidates": 0,
                        "growth_anchor_count": 0,
                        "growth_model_type": "no_active_tracks",
                    },
                ),
            )
        ]

    source_rois_by_session: dict[int, list[int]] = {}
    for active_source in active_sources:
        source_rois_by_session.setdefault(int(active_source.source_session), []).append(
            int(active_source.source_roi)
        )
    matrices_by_source_session = {
        int(source_session): _sparse_pair_matrices(
            sessions,
            feature_cache,
            source_session=int(source_session),
            target_session=next_session,
            source_rois=source_rois,
            edge_top_k=int(config.edge_top_k),
            config=config,
            track2p_prior_edges=prior_edges,
        )
        for source_session, source_rois in source_rois_by_session.items()
    }
    matrix_diagnostics = _matrix_diagnostics(matrices_by_source_session)
    finite_target_rois = sorted(
        {
            int(target_roi)
            for matrices in matrices_by_source_session.values()
            for target_roi in np.asarray(matrices.target_indices, dtype=int)
            if _cell_probability(sessions, next_session, int(target_roi))
            >= float(feature_cache.cell_probability_threshold)
        }
    )
    active_rows = [int(active_source.row_index) for active_source in active_sources]
    gap_active_tracks = sum(1 for active in active_sources if int(active.gap_length) > 0)
    max_gap_length = max((int(active.gap_length) for active in active_sources), default=0)
    row_non_assignment_costs = np.asarray(
        [
            _miss_cost(
                active_source,
                target_session=next_session,
                track2p_prior_edges=prior_edges,
                config=config,
            )
            for active_source in active_sources
        ],
        dtype=float,
    )
    all_miss_cost = float(np.sum(row_non_assignment_costs))
    if not finite_target_rois:
        carried = tracks.copy()
        carried[active_rows, next_session] = -1
        return [
            _MHTHypothesis(
                carried,
                hypothesis.score - all_miss_cost,
                hypothesis.history
                + (
                    {
                        "session_index": int(session_index),
                        "scan_cost": all_miss_cost,
                        "assigned_edges": 0,
                        "missed_tracks": int(len(active_sources)),
                        "gap_active_tracks": int(gap_active_tracks),
                        "gap_reactivated_tracks": 0,
                        "max_gap_length": int(max_gap_length),
                        "scan_candidates": 0,
                        **matrix_diagnostics,
                    },
                ),
            )
        ]

    source_lookup_by_session = {
        int(source_session): {
            int(roi): idx for idx, roi in enumerate(matrices.source_indices)
        }
        for source_session, matrices in matrices_by_source_session.items()
    }
    target_lookup_by_session = {
        int(source_session): {
            int(roi): idx for idx, roi in enumerate(matrices.target_indices)
        }
        for source_session, matrices in matrices_by_source_session.items()
    }
    cost_matrix = np.full(
        (len(active_sources), len(finite_target_rois)), np.inf, dtype=float
    )
    candidate_count = 0
    for row_pos, active_source in enumerate(active_sources):
        source_session = int(active_source.source_session)
        matrices = matrices_by_source_session[source_session]
        source_lookup = source_lookup_by_session[source_session]
        target_lookup = target_lookup_by_session[source_session]
        source_local = source_lookup.get(int(active_source.source_roi))
        if source_local is None:
            continue
        row_scores: list[tuple[float, int]] = []
        for compact_col, target_roi in enumerate(finite_target_rois):
            target_local = target_lookup.get(int(target_roi))
            if target_local is None:
                continue
            score = _edge_score(
                sessions,
                matrices,
                target_session=next_session,
                source_local=int(source_local),
                target_local=int(target_local),
                config=config,
                track2p_prior_edges=prior_edges,
            )
            if int(active_source.gap_length) > 0:
                score -= float(config.gap_reactivation_cost) * float(
                    active_source.gap_length
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
                hypothesis.score - all_miss_cost,
                hypothesis.history
                + (
                    {
                        "session_index": int(session_index),
                        "scan_cost": all_miss_cost,
                        "assigned_edges": 0,
                        "missed_tracks": int(len(active_sources)),
                        "gap_active_tracks": int(gap_active_tracks),
                        "gap_reactivated_tracks": 0,
                        "max_gap_length": int(max_gap_length),
                        "scan_candidates": int(candidate_count),
                        **matrix_diagnostics,
                    },
                ),
            )
        ]

    solutions = murty_k_best_assignments(
        cost_matrix,
        k=max(1, int(config.scan_hypotheses)),
        row_non_assignment_costs=row_non_assignment_costs,
        col_non_assignment_costs=np.zeros((len(finite_target_rois),), dtype=float),
    )
    output: list[_MHTHypothesis] = []
    for solution in solutions:
        assignment = np.asarray(solution["assignment"], dtype=int)
        updated = tracks.copy()
        assigned_edges = 0
        missed_tracks = 0
        gap_reactivated_tracks = 0
        selected_edge_summaries: list[str] = []
        selected_prior_edges = 0
        selected_non_prior_edges = 0
        missed_prior_successors = 0
        for row_pos, active_source in enumerate(active_sources):
            compact_col = int(assignment[int(row_pos)])
            row_index = int(active_source.row_index)
            if compact_col >= 0:
                updated[row_index, next_session] = int(finite_target_rois[compact_col])
                assigned_edges += 1
                edge_summary = _selected_edge_summary(
                    sessions,
                    matrices_by_source_session[int(active_source.source_session)],
                    active_source=active_source,
                    target_session=next_session,
                    target_roi=int(finite_target_rois[compact_col]),
                    config=config,
                    track2p_prior_edges=prior_edges,
                )
                selected_edge_summaries.append(str(edge_summary["summary"]))
                if int(edge_summary["is_track2p_prior"]):
                    selected_prior_edges += 1
                else:
                    selected_non_prior_edges += 1
                if int(active_source.gap_length) > 0:
                    gap_reactivated_tracks += 1
            else:
                updated[row_index, next_session] = -1
                missed_tracks += 1
                if _has_prior_successor(
                    active_source,
                    target_session=next_session,
                    track2p_prior_edges=prior_edges,
                ):
                    missed_prior_successors += 1
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
                        "selected_prior_edges": int(selected_prior_edges),
                        "selected_non_prior_edges": int(selected_non_prior_edges),
                        "missed_prior_successors": int(missed_prior_successors),
                        "selected_edge_summaries": ";".join(selected_edge_summaries),
                        "gap_active_tracks": int(gap_active_tracks),
                        "gap_reactivated_tracks": int(gap_reactivated_tracks),
                        "max_gap_length": int(max_gap_length),
                        "scan_candidates": int(candidate_count),
                        **matrix_diagnostics,
                    },
                ),
            )
        )
    return output


def _edge_score(
    sessions: Sequence[Any],
    matrices: _FullMHTPairMatrices,
    *,
    target_session: int,
    source_local: int,
    target_local: int,
    config: FullMHTConfig,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
) -> float:
    registered = _finite_float(matrices.registered_iou[source_local, target_local], 0.0)
    shifted = _finite_float(matrices.shifted_iou[source_local, target_local], 0.0)
    centroid = _finite_float(matrices.centroid_distance[source_local, target_local], 1e3)
    area_ratio = _finite_float(matrices.area_ratio[source_local, target_local], 0.0)
    growth_residual = _finite_float(
        matrices.growth_residual[source_local, target_local], 0.0
    )
    growth_mahalanobis = _finite_float(
        matrices.growth_mahalanobis[source_local, target_local], 0.0
    )
    local_deformation = _finite_float(
        matrices.local_deformation[source_local, target_local], 0.0
    )
    source_roi = int(matrices.source_indices[int(source_local)])
    target_roi = int(matrices.target_indices[int(target_local)])
    cell_b = _cell_probability(sessions, int(target_session), target_roi)
    threshold_margin = registered - float(matrices.threshold)
    track2p_prior = (
        int(matrices.source_session),
        int(matrices.target_session),
        source_roi,
        target_roi,
    ) in track2p_prior_edges
    score = 0.0
    score += float(config.registered_iou_weight) * registered
    score += float(config.shifted_iou_weight) * shifted
    score += float(config.area_ratio_weight) * max(0.0, min(1.0, area_ratio))
    score += float(config.cell_probability_weight) * max(0.0, min(1.0, cell_b))
    score += float(config.threshold_margin_weight) * max(0.0, threshold_margin)
    score -= float(config.centroid_distance_weight) * max(0.0, centroid)
    score -= float(config.growth_residual_weight) * max(0.0, growth_residual)
    score -= float(config.growth_mahalanobis_weight) * max(0.0, growth_mahalanobis)
    score -= float(config.local_deformation_weight) * max(0.0, local_deformation)
    if track2p_prior:
        score += float(config.track2p_prior_weight)
    elif track2p_prior_edges:
        score -= float(config.track2p_non_prior_penalty)
    return float(score)


def _selected_edge_summary(
    sessions: Sequence[Any],
    matrices: _FullMHTPairMatrices,
    *,
    active_source: _ActiveTrackSource,
    target_session: int,
    target_roi: int,
    config: FullMHTConfig,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
) -> dict[str, Any]:
    source_roi = int(active_source.source_roi)
    target = int(target_roi)
    source_matches = np.flatnonzero(np.asarray(matrices.source_indices) == source_roi)
    target_matches = np.flatnonzero(np.asarray(matrices.target_indices) == target)
    edge = (int(matrices.source_session), int(target_session), source_roi, target)
    if source_matches.size == 0 or target_matches.size == 0:
        is_prior = edge in track2p_prior_edges
        return {
            "is_track2p_prior": int(is_prior),
            "summary": (
                f"{int(matrices.source_session)}:{source_roi}->{int(target_session)}:{target}"
                f"|prior={int(is_prior)}|missing_features=1"
            ),
        }
    source_local = int(source_matches[0])
    target_local = int(target_matches[0])
    registered = _finite_float(matrices.registered_iou[source_local, target_local], 0.0)
    shifted = _finite_float(matrices.shifted_iou[source_local, target_local], 0.0)
    growth_residual = _finite_float(
        matrices.growth_residual[source_local, target_local], 0.0
    )
    growth_mahalanobis = _finite_float(
        matrices.growth_mahalanobis[source_local, target_local], 0.0
    )
    local_deformation = _finite_float(
        matrices.local_deformation[source_local, target_local], 0.0
    )
    cell_probability = _cell_probability(sessions, int(target_session), target)
    score = _edge_score(
        sessions,
        matrices,
        target_session=int(target_session),
        source_local=source_local,
        target_local=target_local,
        config=config,
        track2p_prior_edges=track2p_prior_edges,
    )
    is_prior = edge in track2p_prior_edges
    summary = (
        f"{int(matrices.source_session)}:{source_roi}->{int(target_session)}:{target}"
        f"|prior={int(is_prior)}"
        f"|score={_diagnostic_float(score)}"
        f"|reg={_diagnostic_float(registered)}"
        f"|shift={_diagnostic_float(shifted)}"
        f"|growth={_diagnostic_float(growth_residual)}"
        f"|mahal={_diagnostic_float(growth_mahalanobis)}"
        f"|local={_diagnostic_float(local_deformation)}"
        f"|cell={_diagnostic_float(cell_probability)}"
    )
    return {"is_track2p_prior": int(is_prior), "summary": summary}


def _diagnostic_float(value: float) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.4g}"


def _miss_cost(
    active_source: _ActiveTrackSource,
    *,
    target_session: int,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
    config: FullMHTConfig,
) -> float:
    cost = float(config.miss_cost)
    if not track2p_prior_edges or float(config.track2p_prior_miss_penalty) <= 0.0:
        return cost
    if _has_prior_successor(
        active_source,
        target_session=target_session,
        track2p_prior_edges=track2p_prior_edges,
    ):
        cost += float(config.track2p_prior_miss_penalty)
    return cost


def _has_prior_successor(
    active_source: _ActiveTrackSource,
    *,
    target_session: int,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
) -> bool:
    return any(
        int(session_a) == int(active_source.source_session)
        and int(session_b) == int(target_session)
        and int(roi_a) == int(active_source.source_roi)
        for session_a, session_b, roi_a, _roi_b in track2p_prior_edges
    )


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
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--gap-reactivation-cost", type=float, default=1.0)
    parser.add_argument("--min-output-observations", type=int, default=1)
    parser.add_argument("--min-edge-score", type=float, default=0.25)
    parser.add_argument(
        "--seed-source",
        choices=("reference", "all-cells", "track2p-output"),
        default="reference",
    )
    parser.add_argument("--max-seed-tracks", type=int, default=None)
    parser.add_argument("--registered-iou-weight", type=float, default=1.0)
    parser.add_argument("--shifted-iou-weight", type=float, default=1.5)
    parser.add_argument("--area-ratio-weight", type=float, default=0.25)
    parser.add_argument("--cell-probability-weight", type=float, default=0.25)
    parser.add_argument("--centroid-distance-weight", type=float, default=0.05)
    parser.add_argument("--threshold-margin-weight", type=float, default=0.50)
    parser.add_argument("--growth-residual-weight", type=float, default=0.10)
    parser.add_argument("--growth-mahalanobis-weight", type=float, default=0.25)
    parser.add_argument("--local-deformation-weight", type=float, default=0.50)
    parser.add_argument("--track2p-prior-weight", type=float, default=0.0)
    parser.add_argument("--track2p-non-prior-penalty", type=float, default=0.0)
    parser.add_argument("--track2p-prior-miss-penalty", type=float, default=0.0)
    parser.add_argument("--growth-anchor-min-registered-iou", type=float, default=0.55)
    parser.add_argument("--growth-anchor-min-shifted-iou", type=float, default=0.30)
    parser.add_argument(
        "--growth-anchor-min-cell-probability", type=float, default=0.80
    )
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
            max_gap=max(0, int(args.max_gap)),
            gap_reactivation_cost=float(args.gap_reactivation_cost),
            min_output_observations=max(1, int(args.min_output_observations)),
            min_edge_score=float(args.min_edge_score),
            seed_source=cast(SeedSource, args.seed_source),
            max_seed_tracks=args.max_seed_tracks,
            registered_iou_weight=float(args.registered_iou_weight),
            shifted_iou_weight=float(args.shifted_iou_weight),
            area_ratio_weight=float(args.area_ratio_weight),
            cell_probability_weight=float(args.cell_probability_weight),
            centroid_distance_weight=float(args.centroid_distance_weight),
            threshold_margin_weight=float(args.threshold_margin_weight),
            growth_residual_weight=float(args.growth_residual_weight),
            growth_mahalanobis_weight=float(args.growth_mahalanobis_weight),
            local_deformation_weight=float(args.local_deformation_weight),
            track2p_prior_weight=float(args.track2p_prior_weight),
            track2p_non_prior_penalty=float(args.track2p_non_prior_penalty),
            track2p_prior_miss_penalty=float(args.track2p_prior_miss_penalty),
            growth_anchor_min_registered_iou=float(
                args.growth_anchor_min_registered_iou
            ),
            growth_anchor_min_shifted_iou=float(args.growth_anchor_min_shifted_iou),
            growth_anchor_min_cell_probability=float(
                args.growth_anchor_min_cell_probability
            ),
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
