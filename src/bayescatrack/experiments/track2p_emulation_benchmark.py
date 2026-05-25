"""Track2p-policy emulation using BayesCaTrack loading and scoring.

This diagnostic intentionally mirrors Track2p's core matching policy:

* filter Suite2p ROIs by the configured iscell threshold,
* register consecutive session pairs, plus direct skip pairs when gap rescue is enabled,
* solve a Hungarian assignment on ``1 - IoU``,
* threshold assigned IoUs per session pair with Otsu/minimum thresholding,
* greedily propagate tracks from the first session and stop at the first miss.

The goal is not to replace Track2p. It isolates whether BayesCaTrack's remaining
gap is caused by data loading/registration/mask geometry or by the global solver.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.benchmark_comparison import (
    aggregate_rows,
    write_comparison,
)
from bayescatrack.experiments.track2p_benchmark import (
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    run_track2p_benchmark,
    write_results,
)
from bayescatrack.track2p_registration import register_plane_pair
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.filters import threshold_minimum, threshold_otsu

ThresholdMethod = Literal["otsu", "min"]


def run_track2p_emulation_benchmark(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = "otsu",
    iou_distance_threshold: float = 16.0,
    max_gap: int = 1,
) -> list[SubjectBenchmarkResult]:
    """Run Track2p baseline and Track2p-policy emulation rows."""

    emulation_config = replace(
        config,
        method="global-assignment",
        include_non_cells=False,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=False,
        weighted_centroids=False,
        exclude_overlapping_pixels=False,
        max_gap=int(max_gap),
    )
    baseline_config = replace(emulation_config, method="track2p-baseline")

    results = list(run_track2p_benchmark(baseline_config))
    subject_dirs = discover_subject_dirs(emulation_config.data)
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=emulation_config.data, config=emulation_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=emulation_config
        )
        _validate_reference_roi_indices(
            reference, _load_subject_sessions(subject_dir, emulation_config)
        )
        sessions = _load_subject_sessions(subject_dir, emulation_config)
        predicted = emulate_track2p_tracks(
            sessions,
            transform_type=emulation_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=iou_distance_threshold,
            max_gap=emulation_config.max_gap,
        )
        scores = _score_prediction_against_reference(
            predicted, reference, config=emulation_config
        )
        scores = {
            **scores,
            "track2p_policy_max_gap": int(emulation_config.max_gap),
        }
        variant = f"Track2p-policy emulation ({threshold_method})"
        if emulation_config.max_gap > 1:
            variant = f"{variant} gap-rescue-{emulation_config.max_gap}"
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=variant,
                method="global-assignment",
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
    return results


def emulate_track2p_tracks(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str = "affine",
    threshold_method: ThresholdMethod = "otsu",
    iou_distance_threshold: float = 16.0,
    max_gap: int = 1,
) -> np.ndarray:
    """Return Suite2p-indexed tracks from Track2p's consecutive-link policy.

    ``max_gap`` enables conservative gap rescue.  The propagation still prefers
    the consecutive Track2p-style link whenever it exists, but it may jump over
    isolated missing intermediate sessions when a direct threshold-accepted link
    to a later session is available.  Direct skip links use a proportionally
    larger centroid-distance gate because drift and registration error can
    accumulate across skipped sessions.
    """

    sessions = tuple(sessions)
    max_gap = int(max_gap)
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")

    if not sessions:
        return np.zeros((0, 0), dtype=int)
    if len(sessions) == 1:
        roi_indices = _roi_indices(sessions[0])
        return roi_indices.reshape(-1, 1)

    thresholded_links = _thresholded_links_by_gap(
        sessions,
        transform_type=transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=iou_distance_threshold,
        max_gap=max_gap,
    )

    local_tracks = np.full(
        (sessions[0].plane_data.n_rois, len(sessions)), -1, dtype=int
    )
    seed_rois = _seed_rois_with_outgoing_links(
        thresholded_links,
        max_gap=max_gap,
        first_session_size=sessions[0].plane_data.n_rois,
    )
    if seed_rois.size:
        local_tracks[seed_rois, 0] = seed_rois

    for row_index in range(local_tracks.shape[0]):
        current_session = 0
        current = local_tracks[row_index, current_session]
        if current < 0:
            continue

        while current_session < len(sessions) - 1:
            found_link = False
            max_step = min(max_gap, len(sessions) - 1 - current_session)
            for step in range(1, max_step + 1):
                links = thresholded_links.get((current_session, step))
                if links is None or not links.size:
                    continue
                matches = np.flatnonzero(links[:, 0] == current)
                if matches.size == 0:
                    continue
                current_session += step
                current = int(links[matches[0], 1])
                local_tracks[row_index, current_session] = current
                found_link = True
                break
            if not found_link:
                break

    suite2p_tracks = np.full_like(local_tracks, -1)
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    for session_index, roi_indices in enumerate(roi_indices_by_session):
        valid = local_tracks[:, session_index] >= 0
        if np.any(valid):
            suite2p_tracks[valid, session_index] = roi_indices[
                local_tracks[valid, session_index]
            ]
    return suite2p_tracks


def _thresholded_links_by_gap(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    max_gap: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Return threshold-accepted links keyed by start session and temporal step.

    The IoU distance gate is scaled by the temporal step for direct skip links:
    a two-session jump may require twice the adjacent-session centroid radius.
    Consecutive links are still preferred during propagation, so this only
    broadens opt-in gap rescue when the adjacent link is unavailable.
    """

    links_by_gap: dict[tuple[int, int], np.ndarray] = {}
    for session_index in range(len(sessions) - 1):
        max_step = min(max_gap, len(sessions) - 1 - session_index)
        for step in range(1, max_step + 1):
            step_distance_threshold = float(iou_distance_threshold) * float(step)
            links_by_gap[(session_index, step)] = _thresholded_hungarian_links(
                sessions[session_index],
                sessions[session_index + step],
                transform_type=transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=step_distance_threshold,
            )
    return links_by_gap


def _seed_rois_with_outgoing_links(
    thresholded_links: Mapping[tuple[int, int], np.ndarray],
    *,
    max_gap: int,
    first_session_size: int,
) -> np.ndarray:
    seeds: set[int] = set()
    for step in range(1, max_gap + 1):
        links = thresholded_links.get((0, step))
        if links is None or not links.size:
            continue
        for source_roi in links[:, 0]:
            if 0 <= int(source_roi) < int(first_session_size):
                seeds.add(int(source_roi))
    return np.asarray(sorted(seeds), dtype=int)


def _thresholded_hungarian_links(
    reference_session: Track2pSession,
    moving_session: Track2pSession,
    *,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> np.ndarray:
    registered = register_plane_pair(
        reference_session.plane_data,
        moving_session.plane_data,
        transform_type=transform_type,
    )
    iou = _track2p_cross_iou_matrix(
        np.asarray(reference_session.plane_data.roi_masks) > 0,
        np.asarray(registered.roi_masks) > 0,
        distance_threshold=float(iou_distance_threshold),
    )
    if iou.size == 0:
        return np.zeros((0, 2), dtype=int)

    row_ind, col_ind = linear_sum_assignment(1.0 - iou)
    assigned_iou = iou[row_ind, col_ind]
    threshold = _threshold_assigned_iou(assigned_iou, method=threshold_method)
    keep = assigned_iou > threshold
    if not np.any(keep):
        return np.zeros((0, 2), dtype=int)
    return np.column_stack((row_ind[keep], col_ind[keep])).astype(int)


def _track2p_cross_iou_matrix(
    reference_masks: np.ndarray,
    moving_masks: np.ndarray,
    *,
    distance_threshold: float,
) -> np.ndarray:
    if reference_masks.shape[0] == 0 or moving_masks.shape[0] == 0:
        return np.zeros((reference_masks.shape[0], moving_masks.shape[0]), dtype=float)
    distances = cdist(_mask_centroids(reference_masks), _mask_centroids(moving_masks))
    output = np.zeros((reference_masks.shape[0], moving_masks.shape[0]), dtype=float)
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
                output[row_index, column_index] = float(intersection) / float(union)
    return output


def _mask_centroids(masks: np.ndarray) -> np.ndarray:
    centroids: list[tuple[float, float]] = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            centroids.append((0.0, 0.0))
        else:
            centroids.append((float(np.mean(ys)), float(np.mean(xs))))
    return np.asarray(centroids, dtype=float)


def _threshold_assigned_iou(
    assigned_iou: np.ndarray, *, method: ThresholdMethod
) -> float:
    values = np.asarray(assigned_iou, dtype=float)
    if values.size == 0:
        return float("inf")
    if np.allclose(values, values[0]):
        value = float(values[0])
        # The caller accepts links with ``assigned_iou > threshold``. Returning
        # the observed value for a degenerate positive distribution would reject
        # every identical match, including perfect IoU=1 links. Nudge only
        # positive thresholds downward; all-zero assignments should stay rejected.
        if value > 0.0:
            return float(np.nextafter(value, -np.inf))
        return value
    if method == "otsu":
        return float(threshold_otsu(values))
    if method == "min":
        positive = values[values > 0]
        if positive.size < 3 or np.allclose(positive, positive[0]):
            return float(threshold_otsu(values))
        try:
            return float(threshold_minimum(positive))
        except RuntimeError:
            return float(threshold_otsu(values))
    raise ValueError(f"Unsupported threshold method: {method!r}")


def _roi_indices(session: Track2pSession) -> np.ndarray:
    roi_indices = session.plane_data.roi_indices
    if roi_indices is None:
        return np.arange(session.plane_data.n_rois, dtype=int)
    return np.asarray(roi_indices, dtype=int)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.track2p_emulation_benchmark",
        description=(
            "Compare Track2p output against a BayesCaTrack Track2p-policy "
            "emulation."
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
    parser.add_argument("--transform-type", default="affine")
    parser.add_argument("--threshold-method", choices=("otsu", "min"), default="otsu")
    parser.add_argument("--iou-distance-threshold", type=float, default=16.0)
    parser.add_argument(
        "--max-gap",
        type=int,
        default=1,
        help=(
            "Maximum session jump for conservative gap rescue. The consecutive "
            "Track2p-policy link is always tried first."
        ),
    )
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-emulation benchmark."""

    args = build_arg_parser().parse_args(argv)
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        plane_name=args.plane_name,
        transform_type=args.transform_type,
        max_gap=args.max_gap,
        include_behavior=False,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    results = run_track2p_emulation_benchmark(
        config,
        threshold_method=args.threshold_method,
        iou_distance_threshold=args.iou_distance_threshold,
        max_gap=args.max_gap,
    )
    rows = [result.to_dict() for result in results]
    write_results(rows, args.output, "csv")
    if args.comparison_output is not None:
        comparison_rows = aggregate_rows(
            [{"approach": str(row["variant"]), **row} for row in rows]
        )
        write_comparison(
            comparison_rows,
            args.comparison_output,
            "md",
            highlight_best=True,
            include_best_summary=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
