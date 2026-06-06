"""Track2pPolicy with a label-free growth-regularized assignment score.

This experimental row moves the lower-left growth/deformation prior earlier in
the pipeline.  Instead of applying a post-hoc growth veto, it fits a per-session
growth field from high-confidence policy edges and adds a small growth penalty
to the Hungarian assignment cost:

``cost = 1 - registered_iou + lambda_growth * capped_mahalanobis + lambda_area * area_residual``

Manual-GT labels are used only by the benchmark scorer, never by the growth
model fitting or assignment.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments import track2p_policy_growth_veto_whatif as veto
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
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_growth_field_residual_audit import (
    _growth_models_by_pair,
    _identity_growth_model,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
    _link_diagnostic,
    _roi_indices,
    _threshold_assigned_iou,
    _track2p_cross_iou_diagnostic_matrices,
    _tracks_from_pair_links,
    emulate_track2p_pruned_tracks,
)
from bayescatrack.track2p_registration import register_plane_pair
from scipy.optimize import linear_sum_assignment

METHOD = "track2p-policy-growth-regularized-assignment"


@dataclass(frozen=True)
class GrowthRegularizationConfig:
    """Fixed growth-assignment penalty settings."""

    lambda_growth: float = 0.10
    lambda_area: float = 0.05
    growth_mahalanobis_cap: float = 30.0
    growth_penalty_min_iou: float = 0.05
    anchor_min_registered_iou: float = 0.50
    anchor_min_shifted_iou: float = 0.0
    anchor_min_cell_probability: float = 0.80


@dataclass(frozen=True)
class GrowthRegularizedAssignmentPrediction:
    """Growth-regularized tracks and accepted-link diagnostics."""

    tracks: np.ndarray
    diagnostics: tuple[Track2pPolicyLinkDiagnostic, ...]
    anchor_counts: Mapping[tuple[int, int], int]


@dataclass(frozen=True)
class _SimpleGrowthContext:
    centroids: tuple[Mapping[int, np.ndarray], ...]
    areas: tuple[Mapping[int, float], ...]


def run_track2p_policy_growth_regularized_assignment_benchmark(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    growth_config: GrowthRegularizationConfig | None = None,
) -> list[SubjectBenchmarkResult]:
    """Run the growth-regularized Track2pPolicy row over all subjects."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    growth_config = growth_config or GrowthRegularizationConfig()
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
        prediction = emulate_track2p_growth_regularized_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            growth_config=growth_config,
        )
        scores = _score_prediction_against_reference(
            prediction.tracks, reference, config=policy_config
        )
        scores = {
            **scores,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_growth_regularized_lambda_growth": float(
                growth_config.lambda_growth
            ),
            "track2p_growth_regularized_lambda_area": float(growth_config.lambda_area),
            "track2p_growth_regularized_mahalanobis_cap": float(
                growth_config.growth_mahalanobis_cap
            ),
            "track2p_growth_regularized_penalty_min_iou": float(
                growth_config.growth_penalty_min_iou
            ),
            "track2p_growth_regularized_anchor_edges": int(
                sum(prediction.anchor_counts.values())
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=(
                    "Track2p-policy growth-regularized "
                    f"lambda={growth_config.lambda_growth:g}"
                ),
                method=cast(Any, METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
    return results


def emulate_track2p_growth_regularized_tracks(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str = TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    growth_config: GrowthRegularizationConfig | None = None,
) -> GrowthRegularizedAssignmentPrediction:
    """Return Suite2p-indexed tracks from growth-regularized adjacent links."""

    sessions = tuple(sessions)
    if not sessions:
        return GrowthRegularizedAssignmentPrediction(
            tracks=np.zeros((0, 0), dtype=int),
            diagnostics=(),
            anchor_counts={},
        )
    if len(sessions) == 1:
        roi_indices = _roi_indices(sessions[0])
        return GrowthRegularizedAssignmentPrediction(
            tracks=roi_indices.reshape(-1, 1),
            diagnostics=(),
            anchor_counts={},
        )

    growth_config = growth_config or GrowthRegularizationConfig()
    baseline = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    anchor_edges = veto._anchor_edges_from_policy_diagnostics(
        sessions,
        feature_cache=None,
        diagnostics=baseline.diagnostics,
        track2p=baseline.tracks,
        component_cleanup=baseline.tracks,
        combined=baseline.tracks,
        min_registered_iou=float(growth_config.anchor_min_registered_iou),
        min_shifted_iou=float(growth_config.anchor_min_shifted_iou),
        min_cell_probability=float(growth_config.anchor_min_cell_probability),
    )
    growth_models = _growth_models_by_pair(sessions, anchor_edges)
    growth_context = _simple_growth_context(sessions)

    pair_links: list[np.ndarray] = []
    diagnostics: list[Track2pPolicyLinkDiagnostic] = []
    for index in range(len(sessions) - 1):
        links, pair_diagnostics = _thresholded_growth_regularized_links(
            sessions[index],
            sessions[index + 1],
            session_index=index,
            transform_type=transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            growth_config=growth_config,
            growth_context=growth_context,
            growth_model=growth_models.get(
                (index, index + 1), _identity_growth_model()
            ),
        )
        pair_links.append(links)
        diagnostics.extend(pair_diagnostics)
    return GrowthRegularizedAssignmentPrediction(
        tracks=_tracks_from_pair_links(sessions, pair_links),
        diagnostics=tuple(diagnostics),
        anchor_counts={pair: len(edges) for pair, edges in anchor_edges.items()},
    )


def _simple_growth_context(
    sessions: Sequence[Track2pSession],
) -> _SimpleGrowthContext:
    """Return only the ROI centroid/area lookup needed by assignment costs."""

    all_centroids: list[dict[int, np.ndarray]] = []
    all_areas: list[dict[int, float]] = []
    for session in sessions:
        roi_indices = _roi_indices(session)
        masks = np.asarray(session.plane_data.roi_masks) > 0
        centroids: dict[int, np.ndarray] = {}
        areas: dict[int, float] = {}
        for local_index, suite2p_roi in enumerate(roi_indices):
            mask = masks[int(local_index)]
            area = float(mask.sum())
            areas[int(suite2p_roi)] = area
            if area <= 0.0:
                continue
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                continue
            centroids[int(suite2p_roi)] = np.asarray(
                [float(np.mean(xs)), float(np.mean(ys))],
                dtype=float,
            )
        all_centroids.append(centroids)
        all_areas.append(areas)
    return _SimpleGrowthContext(
        centroids=tuple(all_centroids),
        areas=tuple(all_areas),
    )


def growth_regularized_cost_matrix(
    registered_iou: np.ndarray,
    growth_mahalanobis: np.ndarray,
    area_growth_residual: np.ndarray,
    *,
    lambda_growth: float,
    lambda_area: float,
    growth_mahalanobis_cap: float,
) -> np.ndarray:
    """Return ``1 - IoU`` plus normalized growth/area penalties."""

    iou = np.asarray(registered_iou, dtype=float)
    mahalanobis = np.asarray(growth_mahalanobis, dtype=float)
    area_residual = np.asarray(area_growth_residual, dtype=float)
    capped = np.minimum(
        np.nan_to_num(
            mahalanobis,
            nan=float(growth_mahalanobis_cap),
            posinf=float(growth_mahalanobis_cap),
        ),
        float(growth_mahalanobis_cap),
    )
    normalized_growth = capped / max(float(growth_mahalanobis_cap), 1.0e-12)
    area_penalty = np.nan_to_num(area_residual, nan=1.0, posinf=1.0)
    return (
        1.0
        - iou
        + float(lambda_growth) * normalized_growth
        + float(lambda_area) * area_penalty
    )


def _thresholded_growth_regularized_links(
    reference_session: Track2pSession,
    moving_session: Track2pSession,
    *,
    session_index: int,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    growth_config: GrowthRegularizationConfig,
    growth_context: Any,
    growth_model: Any,
) -> tuple[np.ndarray, tuple[Track2pPolicyLinkDiagnostic, ...]]:
    registered = register_plane_pair(
        reference_session.plane_data,
        moving_session.plane_data,
        transform_type=transform_type,
    )
    iou, distances, area_ratios = _track2p_cross_iou_diagnostic_matrices(
        np.asarray(reference_session.plane_data.roi_masks) > 0,
        np.asarray(registered.roi_masks) > 0,
        distance_threshold=float(iou_distance_threshold),
    )
    growth_mahalanobis, area_growth = _growth_penalty_matrices(
        iou.shape,
        session_index=session_index,
        source_indices=_roi_indices(reference_session),
        target_indices=_roi_indices(moving_session),
        growth_context=growth_context,
        growth_model=growth_model,
        candidate_mask=iou > float(growth_config.growth_penalty_min_iou),
        growth_mahalanobis_cap=float(growth_config.growth_mahalanobis_cap),
    )
    cost = growth_regularized_cost_matrix(
        iou,
        growth_mahalanobis,
        area_growth,
        lambda_growth=float(growth_config.lambda_growth),
        lambda_area=float(growth_config.lambda_area),
        growth_mahalanobis_cap=float(growth_config.growth_mahalanobis_cap),
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_iou = iou[row_ind, col_ind]
    threshold = _threshold_assigned_iou(assigned_iou, method=threshold_method)
    kept_links: list[tuple[int, int]] = []
    diagnostics: list[Track2pPolicyLinkDiagnostic] = []
    for row_index, col_index, value in zip(row_ind, col_ind, assigned_iou, strict=True):
        if float(value) <= float(threshold):
            continue
        row = int(row_index)
        col = int(col_index)
        diagnostic = _link_diagnostic(
            iou,
            row=row,
            col=col,
            assigned_iou=float(value),
            threshold=threshold,
            session_index=session_index,
            distance=float(distances[row, col]),
            area_ratio=float(area_ratios[row, col]),
            prune_config=_no_prune_config(),
        )
        diagnostics.append(diagnostic)
        kept_links.append((row, col))
    return np.asarray(kept_links, dtype=int).reshape(-1, 2), tuple(diagnostics)


def _growth_penalty_matrices(
    shape: tuple[int, int],
    *,
    session_index: int,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    growth_context: Any,
    growth_model: Any,
    candidate_mask: np.ndarray,
    growth_mahalanobis_cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    mahalanobis = np.full(shape, float(growth_mahalanobis_cap), dtype=float)
    area_residual = np.ones(shape, dtype=float)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    rows, cols = np.nonzero(candidate_mask)
    if rows.size == 0:
        return mahalanobis, area_residual

    source_centroids = growth_context.centroids[int(session_index)]
    target_centroids = growth_context.centroids[int(session_index) + 1]
    source_areas = growth_context.areas[int(session_index)]
    target_areas = growth_context.areas[int(session_index) + 1]
    source_indices = np.asarray(source_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)

    source_xy = np.full((len(source_indices), 2), np.nan, dtype=float)
    target_xy = np.full((len(target_indices), 2), np.nan, dtype=float)
    source_area = np.full(len(source_indices), np.nan, dtype=float)
    target_area = np.full(len(target_indices), np.nan, dtype=float)
    for row, roi in enumerate(source_indices):
        source = source_centroids.get(int(roi))
        if source is not None:
            source_xy[int(row)] = np.asarray(source, dtype=float)
        source_area[int(row)] = float(source_areas.get(int(roi), float("nan")))
    for col, roi in enumerate(target_indices):
        target = target_centroids.get(int(roi))
        if target is not None:
            target_xy[int(col)] = np.asarray(target, dtype=float)
        target_area[int(col)] = float(target_areas.get(int(roi), float("nan")))

    sources = source_xy[rows]
    targets = target_xy[cols]
    valid = np.isfinite(sources).all(axis=1) & np.isfinite(targets).all(axis=1)
    if not np.any(valid):
        return mahalanobis, area_residual

    valid_rows = rows[valid]
    valid_cols = cols[valid]
    sources = sources[valid]
    targets = targets[valid]
    affine_xy = np.asarray(growth_model.affine_xy, dtype=float)
    predicted = sources @ affine_xy[:, :2].T + affine_xy[:, 2]
    residual = targets - predicted
    covariance_inverse = np.asarray(growth_model.covariance_inverse, dtype=float)
    distances = np.sqrt(
        np.maximum(
            0.0,
            np.einsum("ij,jk,ik->i", residual, covariance_inverse, residual),
        )
    )
    area_a_array = source_area[valid_rows]
    area_b_array = target_area[valid_cols]
    observed = np.divide(
        area_b_array,
        area_a_array,
        out=np.full_like(area_b_array, np.nan, dtype=float),
        where=np.isfinite(area_a_array) & (area_a_array > 0.0),
    )
    expected = float(growth_model.expected_area_ratio)
    area_values = np.abs(np.log(observed / expected))
    area_values[~np.isfinite(area_values)] = 1.0
    mahalanobis[valid_rows, valid_cols] = distances
    area_residual[valid_rows, valid_cols] = area_values
    return mahalanobis, area_residual


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the growth-regularized assignment parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-growth-regularized-assignment",
        description="Run Track2pPolicy with a label-free growth penalty in Hungarian assignment.",
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
        "--iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument("--lambda-growth", type=float, default=0.10)
    parser.add_argument("--lambda-area", type=float, default=0.05)
    parser.add_argument("--growth-mahalanobis-cap", type=float, default=30.0)
    parser.add_argument("--growth-penalty-min-iou", type=float, default=0.05)
    parser.add_argument("--anchor-min-registered-iou", type=float, default=0.50)
    parser.add_argument("--anchor-min-shifted-iou", type=float, default=0.0)
    parser.add_argument("--anchor-min-cell-probability", type=float, default=0.80)
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
    """Run the growth-regularized assignment benchmark CLI."""

    args = build_arg_parser().parse_args(argv)
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
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=False,
        weighted_centroids=False,
        exclude_overlapping_pixels=False,
    )
    results = run_track2p_policy_growth_regularized_assignment_benchmark(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        growth_config=GrowthRegularizationConfig(
            lambda_growth=float(args.lambda_growth),
            lambda_area=float(args.lambda_area),
            growth_mahalanobis_cap=float(args.growth_mahalanobis_cap),
            growth_penalty_min_iou=float(args.growth_penalty_min_iou),
            anchor_min_registered_iou=float(args.anchor_min_registered_iou),
            anchor_min_shifted_iou=float(args.anchor_min_shifted_iou),
            anchor_min_cell_probability=float(args.anchor_min_cell_probability),
        ),
    )
    rows = [result.to_dict() for result in results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
