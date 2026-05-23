"""Conservative prune-only variant of the promoted Track2p-policy row.

The plain Track2p-policy row is intentionally close to Track2p and currently is
BayesCaTrack's strongest Track2p-style result.  This module adds an opt-in
post-filter that can only remove accepted policy edges.  It never rescues new
edges, which keeps the experiment aligned with the next result target: reduce
extra policy false positives while preserving the complete-track advantage.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
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
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
    track2p_policy_config,
)
from bayescatrack.track2p_registration import register_plane_pair
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.filters import threshold_minimum, threshold_otsu

TRACK2P_POLICY_PRUNED_METHOD = "track2p-policy-pruned"


@dataclass(frozen=True)
class Track2pPolicyPruneConfig:
    """Conservative edge-removal thresholds for Track2p-policy links.

    A link is pruned only when all three signals are weak:

    * the accepted IoU is barely above the Track2p-policy threshold,
    * both the row and column have close competing alternatives,
    * shape/geometry evidence is poor.

    The defaults are intentionally cautious.  Treat them as a starting point for
    fold-clean tuning, not as a claim that they are globally optimal.
    """

    threshold_margin: float = 0.02
    competition_margin: float = 0.02
    min_area_ratio: float = 0.45
    centroid_distance: float = 10.0

    def __post_init__(self) -> None:
        _require_nonnegative(self.threshold_margin, name="threshold_margin")
        _require_nonnegative(self.competition_margin, name="competition_margin")
        _require_probability_like(self.min_area_ratio, name="min_area_ratio")
        _require_nonnegative(self.centroid_distance, name="centroid_distance")


@dataclass(frozen=True)
class Track2pPolicyLinkDiagnostic:
    """Diagnostics for one threshold-accepted policy link."""

    session_index: int
    local_roi_a: int
    local_roi_b: int
    assigned_iou: float
    threshold: float
    threshold_margin: float
    row_margin: float
    column_margin: float
    centroid_distance: float
    area_ratio: float
    pruned: bool
    prune_reason: str


@dataclass(frozen=True)
class Track2pPolicyPrunedPrediction:
    """Pruned track matrix plus edge diagnostics."""

    tracks: np.ndarray
    diagnostics: tuple[Track2pPolicyLinkDiagnostic, ...]


@dataclass(frozen=True)
class Track2pPolicyPrunedBenchmarkOutput:
    """Benchmark results and optional per-edge diagnostic rows."""

    results: tuple[SubjectBenchmarkResult, ...]
    diagnostic_rows: tuple[dict[str, float | int | str], ...]


def run_track2p_policy_pruned_benchmark(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    prune_config: Track2pPolicyPruneConfig | None = None,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
) -> list[SubjectBenchmarkResult]:
    """Run the prune-only Track2p-policy benchmark row."""

    return list(
        run_track2p_policy_pruned_benchmark_with_diagnostics(
            config,
            threshold_method=threshold_method,
            iou_distance_threshold=iou_distance_threshold,
            prune_config=prune_config,
            transform_type=transform_type,
            cell_probability_threshold=cell_probability_threshold,
        ).results
    )


def run_track2p_policy_pruned_benchmark_with_diagnostics(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    prune_config: Track2pPolicyPruneConfig | None = None,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
) -> Track2pPolicyPrunedBenchmarkOutput:
    """Run the prune-only policy row and retain edge diagnostics."""

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

    prune_config = prune_config or Track2pPolicyPruneConfig()
    results: list[SubjectBenchmarkResult] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=prune_config,
        )
        scores = _score_prediction_against_reference(
            prediction.tracks, reference, config=policy_config
        )
        pruned_edges = int(sum(diagnostic.pruned for diagnostic in prediction.diagnostics))
        candidate_edges = int(len(prediction.diagnostics))
        scores = {
            **scores,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_policy_prune_threshold_margin": float(
                prune_config.threshold_margin
            ),
            "track2p_policy_prune_competition_margin": float(
                prune_config.competition_margin
            ),
            "track2p_policy_prune_min_area_ratio": float(
                prune_config.min_area_ratio
            ),
            "track2p_policy_prune_centroid_distance": float(
                prune_config.centroid_distance
            ),
            "track2p_policy_candidate_edges": candidate_edges,
            "track2p_policy_pruned_edges": pruned_edges,
            "track2p_policy_kept_edges": candidate_edges - pruned_edges,
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=f"Track2p-policy pruned {threshold_method}",
                method=cast(Any, TRACK2P_POLICY_PRUNED_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        diagnostic_rows.extend(
            diagnostic_rows_for_subject(
                subject_dir.name,
                sessions,
                prediction.diagnostics,
                metadata={
                    "threshold_method": str(threshold_method),
                    "iou_distance_threshold": float(iou_distance_threshold),
                    "cell_probability_threshold": float(
                        policy_config.cell_probability_threshold
                    ),
                    "transform_type": str(policy_config.transform_type),
                    "prune_threshold_margin": float(prune_config.threshold_margin),
                    "prune_competition_margin": float(
                        prune_config.competition_margin
                    ),
                    "prune_min_area_ratio": float(prune_config.min_area_ratio),
                    "prune_centroid_distance": float(
                        prune_config.centroid_distance
                    ),
                },
            )
        )
    return Track2pPolicyPrunedBenchmarkOutput(
        results=tuple(results), diagnostic_rows=tuple(diagnostic_rows)
    )


def emulate_track2p_pruned_tracks(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str = TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    prune_config: Track2pPolicyPruneConfig | None = None,
) -> Track2pPolicyPrunedPrediction:
    """Return Track2p-policy tracks after conservative post-threshold pruning."""

    sessions = tuple(sessions)
    if not sessions:
        return Track2pPolicyPrunedPrediction(
            tracks=np.zeros((0, 0), dtype=int), diagnostics=()
        )
    if len(sessions) == 1:
        roi_indices = _roi_indices(sessions[0])
        return Track2pPolicyPrunedPrediction(
            tracks=roi_indices.reshape(-1, 1), diagnostics=()
        )

    prune_config = prune_config or Track2pPolicyPruneConfig()
    pair_links: list[np.ndarray] = []
    diagnostics: list[Track2pPolicyLinkDiagnostic] = []
    for index in range(len(sessions) - 1):
        links, pair_diagnostics = _thresholded_pruned_hungarian_links(
            sessions[index],
            sessions[index + 1],
            session_index=index,
            transform_type=transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=prune_config,
        )
        pair_links.append(links)
        diagnostics.extend(pair_diagnostics)
    return Track2pPolicyPrunedPrediction(
        tracks=_tracks_from_pair_links(sessions, pair_links),
        diagnostics=tuple(diagnostics),
    )


def policy_link_diagnostics_from_iou_matrix(
    iou_matrix: Any,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    prune_config: Track2pPolicyPruneConfig | None = None,
    session_index: int = 0,
    distances: Any | None = None,
    area_ratios: Any | None = None,
    threshold_override: float | None = None,
) -> tuple[np.ndarray, tuple[Track2pPolicyLinkDiagnostic, ...]]:
    """Return kept local links and diagnostics from an IoU matrix.

    This pure helper is useful for focused tests and for future LOSO tuning code.
    Production code passes the IoU, distance, and area-ratio matrices produced
    from registered ROI masks.
    """

    iou = np.asarray(iou_matrix, dtype=float)
    if iou.ndim != 2:
        raise ValueError("iou_matrix must be two-dimensional")
    if iou.size == 0:
        return np.zeros((0, 2), dtype=int), ()
    distance_matrix = _default_matrix_like(
        distances, shape=iou.shape, default=0.0, name="distances"
    )
    area_ratio_matrix = _default_matrix_like(
        area_ratios, shape=iou.shape, default=1.0, name="area_ratios"
    )
    prune_config = prune_config or Track2pPolicyPruneConfig()

    row_ind, col_ind = linear_sum_assignment(1.0 - iou)
    assigned_iou = iou[row_ind, col_ind]
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else _threshold_assigned_iou(assigned_iou, method=threshold_method)
    )
    threshold_keep = assigned_iou > threshold
    kept_links: list[tuple[int, int]] = []
    diagnostics: list[Track2pPolicyLinkDiagnostic] = []
    for row_index, col_index, value, keep in zip(
        row_ind, col_ind, assigned_iou, threshold_keep, strict=True
    ):
        if not bool(keep):
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
            distance=float(distance_matrix[row, col]),
            area_ratio=float(area_ratio_matrix[row, col]),
            prune_config=prune_config,
        )
        diagnostics.append(diagnostic)
        if not diagnostic.pruned:
            kept_links.append((row, col))
    return np.asarray(kept_links, dtype=int).reshape(-1, 2), tuple(diagnostics)


def should_prune_policy_edge(
    *,
    assigned_iou: float,
    threshold: float,
    row_margin: float,
    column_margin: float,
    area_ratio: float,
    centroid_distance: float,
    config: Track2pPolicyPruneConfig | None = None,
) -> bool:
    """Return whether a threshold-accepted policy edge should be removed."""

    config = config or Track2pPolicyPruneConfig()
    threshold_margin = float(assigned_iou) - float(threshold)
    weak_threshold = threshold_margin <= config.threshold_margin
    weak_competition = (
        float(row_margin) <= config.competition_margin
        and float(column_margin) <= config.competition_margin
    )
    weak_geometry = (
        float(area_ratio) < config.min_area_ratio
        or float(centroid_distance) >= config.centroid_distance
    )
    return bool(weak_threshold and weak_competition and weak_geometry)


def diagnostic_rows_for_subject(
    subject: str,
    sessions: Sequence[Track2pSession],
    diagnostics: Sequence[Track2pPolicyLinkDiagnostic],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, float | int | str]]:
    """Convert local-index diagnostics to Suite2p-indexed CSV/JSON rows."""

    sessions = tuple(sessions)
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    session_names = [session.session_name for session in sessions]
    meta = {key: _format_metadata_value(value) for key, value in dict(metadata or {}).items()}
    rows: list[dict[str, float | int | str]] = []
    for diagnostic in diagnostics:
        session_index = int(diagnostic.session_index)
        rows.append(
            {
                "subject": subject,
                "session_a": session_index,
                "session_b": session_index + 1,
                "session_a_name": str(session_names[session_index]),
                "session_b_name": str(session_names[session_index + 1]),
                "local_roi_a": int(diagnostic.local_roi_a),
                "local_roi_b": int(diagnostic.local_roi_b),
                "suite2p_roi_a": int(
                    roi_indices_by_session[session_index][diagnostic.local_roi_a]
                ),
                "suite2p_roi_b": int(
                    roi_indices_by_session[session_index + 1][diagnostic.local_roi_b]
                ),
                "assigned_iou": float(diagnostic.assigned_iou),
                "threshold": float(diagnostic.threshold),
                "threshold_margin": float(diagnostic.threshold_margin),
                "row_margin": float(diagnostic.row_margin),
                "column_margin": float(diagnostic.column_margin),
                "centroid_distance": float(diagnostic.centroid_distance),
                "area_ratio": float(diagnostic.area_ratio),
                "pruned": int(diagnostic.pruned),
                "prune_reason": diagnostic.prune_reason,
                **meta,
            }
        )
    return rows


def write_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write edge diagnostics as CSV or JSON."""

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
    """Build the command-line parser for the prune-only policy method."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-pruned",
        description="Run a conservative prune-only Track2p-policy benchmark method.",
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
        "--transform-type",
        default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
        help="Registration transform; defaults to the tuned Track2p-policy affine setting.",
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
        "--prune-threshold-margin",
        type=float,
        default=Track2pPolicyPruneConfig().threshold_margin,
        help="Prune only links whose IoU is within this margin above the Track2p threshold.",
    )
    parser.add_argument(
        "--prune-competition-margin",
        type=float,
        default=Track2pPolicyPruneConfig().competition_margin,
        help="Prune only links whose row and column alternatives are this close.",
    )
    parser.add_argument(
        "--prune-min-area-ratio",
        type=float,
        default=Track2pPolicyPruneConfig().min_area_ratio,
        help="Geometry is weak when min(area_a, area_b) / max(area_a, area_b) is below this value.",
    )
    parser.add_argument(
        "--prune-centroid-distance",
        type=float,
        default=Track2pPolicyPruneConfig().centroid_distance,
        help="Geometry is weak when registered centroid distance is at least this many pixels.",
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument(
        "--diagnostics-format", choices=("csv", "json"), default="csv"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the prune-only Track2p-policy benchmark CLI."""

    args = build_arg_parser().parse_args(argv)
    prune_config = Track2pPolicyPruneConfig(
        threshold_margin=args.prune_threshold_margin,
        competition_margin=args.prune_competition_margin,
        min_area_ratio=args.prune_min_area_ratio,
        centroid_distance=args.prune_centroid_distance,
    )
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
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
    output = run_track2p_policy_pruned_benchmark_with_diagnostics(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        prune_config=prune_config,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.diagnostics_output is not None:
        write_diagnostic_rows(
            output.diagnostic_rows,
            args.diagnostics_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    return 0


def _thresholded_pruned_hungarian_links(
    reference_session: Track2pSession,
    moving_session: Track2pSession,
    *,
    session_index: int,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    prune_config: Track2pPolicyPruneConfig,
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
    return policy_link_diagnostics_from_iou_matrix(
        iou,
        threshold_method=threshold_method,
        prune_config=prune_config,
        session_index=session_index,
        distances=distances,
        area_ratios=area_ratios,
    )


def _track2p_cross_iou_diagnostic_matrices(
    reference_masks: np.ndarray,
    moving_masks: np.ndarray,
    *,
    distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (reference_masks.shape[0], moving_masks.shape[0])
    if reference_masks.shape[0] == 0 or moving_masks.shape[0] == 0:
        return (
            np.zeros(shape, dtype=float),
            np.zeros(shape, dtype=float),
            np.ones(shape, dtype=float),
        )
    reference_masks = np.asarray(reference_masks, dtype=bool)
    moving_masks = np.asarray(moving_masks, dtype=bool)
    distances = cdist(_mask_centroids(reference_masks), _mask_centroids(moving_masks))
    area_ratios = _area_ratio_matrix(reference_masks, moving_masks)
    iou = np.zeros(shape, dtype=float)
    for row_index in range(reference_masks.shape[0]):
        for column_index in range(moving_masks.shape[0]):
            if distances[row_index, column_index] > distance_threshold:
                continue
            intersection = np.logical_and(
                reference_masks[row_index], moving_masks[column_index]
            ).sum()
            union = np.logical_or(
                reference_masks[row_index], moving_masks[column_index]
            ).sum()
            if union > 0:
                iou[row_index, column_index] = float(intersection) / float(union)
    return iou, distances, area_ratios


def _link_diagnostic(
    iou: np.ndarray,
    *,
    row: int,
    col: int,
    assigned_iou: float,
    threshold: float,
    session_index: int,
    distance: float,
    area_ratio: float,
    prune_config: Track2pPolicyPruneConfig,
) -> Track2pPolicyLinkDiagnostic:
    row_margin = _margin_against_competitor(iou[row, :], selected_index=col)
    column_margin = _margin_against_competitor(iou[:, col], selected_index=row)
    threshold_margin = float(assigned_iou) - float(threshold)
    pruned = should_prune_policy_edge(
        assigned_iou=assigned_iou,
        threshold=threshold,
        row_margin=row_margin,
        column_margin=column_margin,
        area_ratio=area_ratio,
        centroid_distance=distance,
        config=prune_config,
    )
    return Track2pPolicyLinkDiagnostic(
        session_index=int(session_index),
        local_roi_a=int(row),
        local_roi_b=int(col),
        assigned_iou=float(assigned_iou),
        threshold=float(threshold),
        threshold_margin=float(threshold_margin),
        row_margin=float(row_margin),
        column_margin=float(column_margin),
        centroid_distance=float(distance),
        area_ratio=float(area_ratio),
        pruned=bool(pruned),
        prune_reason=(
            _prune_reason(
                threshold_margin=threshold_margin,
                row_margin=row_margin,
                column_margin=column_margin,
                area_ratio=area_ratio,
                centroid_distance=distance,
                config=prune_config,
            )
            if pruned
            else "kept"
        ),
    )


def _tracks_from_pair_links(
    sessions: Sequence[Track2pSession], pair_links: Sequence[np.ndarray]
) -> np.ndarray:
    local_tracks = np.full(
        (sessions[0].plane_data.n_rois, len(sessions)), -1, dtype=int
    )
    first_links = pair_links[0]
    if first_links.size:
        local_tracks[first_links[:, 0], 0] = first_links[:, 0]

    for row_index in range(local_tracks.shape[0]):
        current = local_tracks[row_index, 0]
        if current < 0:
            continue
        for session_index, links in enumerate(pair_links):
            if not links.size:
                break
            matches = np.flatnonzero(links[:, 0] == current)
            if matches.size == 0:
                break
            current = int(links[matches[0], 1])
            local_tracks[row_index, session_index + 1] = current

    suite2p_tracks = np.full_like(local_tracks, -1)
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    for session_index, roi_indices in enumerate(roi_indices_by_session):
        valid = local_tracks[:, session_index] >= 0
        if np.any(valid):
            suite2p_tracks[valid, session_index] = roi_indices[
                local_tracks[valid, session_index]
            ]
    return suite2p_tracks


def _threshold_assigned_iou(
    assigned_iou: np.ndarray, *, method: ThresholdMethod) -> float:
    values = np.asarray(assigned_iou, dtype=float)
    if values.size == 0:
        return float("inf")
    if np.allclose(values, values[0]):
        return float(values[0])
    if method == "otsu":
        return float(threshold_otsu(values))
    if method == "min":
        positive = values[values > 0]
        if positive.size < 3 or np.allclose(positive, positive[0]):
            return float(threshold_otsu(values))
        return float(threshold_minimum(positive))
    raise ValueError(f"Unsupported threshold method: {method!r}")


def _mask_centroids(masks: np.ndarray) -> np.ndarray:
    centroids: list[tuple[float, float]] = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            centroids.append((0.0, 0.0))
        else:
            centroids.append((float(np.mean(ys)), float(np.mean(xs))))
    return np.asarray(centroids, dtype=float)


def _area_ratio_matrix(reference_masks: np.ndarray, moving_masks: np.ndarray) -> np.ndarray:
    reference_areas = np.asarray(reference_masks, dtype=bool).reshape(
        reference_masks.shape[0], -1
    ).sum(axis=1)
    moving_areas = np.asarray(moving_masks, dtype=bool).reshape(
        moving_masks.shape[0], -1
    ).sum(axis=1)
    ratios = np.ones((reference_masks.shape[0], moving_masks.shape[0]), dtype=float)
    for row_index, reference_area in enumerate(reference_areas):
        for column_index, moving_area in enumerate(moving_areas):
            largest = max(float(reference_area), float(moving_area))
            if largest == 0.0:
                ratios[row_index, column_index] = 1.0
            else:
                ratios[row_index, column_index] = min(
                    float(reference_area), float(moving_area)
                ) / largest
    return ratios


def _margin_against_competitor(values: np.ndarray, *, selected_index: int) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size <= 1:
        return float("inf")
    competitors = np.delete(values, int(selected_index))
    if competitors.size == 0:
        return float("inf")
    return float(values[int(selected_index)] - np.max(competitors))


def _prune_reason(
    *,
    threshold_margin: float,
    row_margin: float,
    column_margin: float,
    area_ratio: float,
    centroid_distance: float,
    config: Track2pPolicyPruneConfig,
) -> str:
    reasons: list[str] = []
    if threshold_margin <= config.threshold_margin:
        reasons.append("weak-threshold-margin")
    if row_margin <= config.competition_margin:
        reasons.append("weak-row-margin")
    if column_margin <= config.competition_margin:
        reasons.append("weak-column-margin")
    if area_ratio < config.min_area_ratio:
        reasons.append("weak-area-ratio")
    if centroid_distance >= config.centroid_distance:
        reasons.append("large-centroid-distance")
    return ";".join(reasons) if reasons else "pruned"


def _default_matrix_like(
    value: Any | None, *, shape: tuple[int, int], default: float, name: str
) -> np.ndarray:
    if value is None:
        return np.full(shape, float(default), dtype=float)
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    return matrix


def _roi_indices(session: Track2pSession) -> np.ndarray:
    roi_indices = session.plane_data.roi_indices
    if roi_indices is None:
        return np.arange(session.plane_data.n_rois, dtype=int)
    return np.asarray(roi_indices, dtype=int)


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _require_nonnegative(value: float, *, name: str) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_probability_like(value: float, *, name: str) -> None:
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
