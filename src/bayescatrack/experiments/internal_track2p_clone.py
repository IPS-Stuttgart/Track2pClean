"""Internal Track2p-style matcher for raw Suite2p/Track2p inputs.

This module intentionally mirrors Track2p's default matching loop instead of
loading precomputed Track2p outputs.  It is a diagnostic benchmark row: the
same BayesCaTrack-loaded subject can now be evaluated with Track2p's
consecutive-session IoU matching, Hungarian assignment, edge-local thresholding,
and first-session propagation logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
from bayescatrack.core.bridge import (
    CalciumPlaneData,
    Track2pSession,
    find_track2p_session_dirs,
    load_raw_npy_plane,
)
from bayescatrack.matching import build_track_rows_from_matches
from bayescatrack.track2p_registration import register_plane_pair

try:  # pragma: no cover - exercised in real runtime/CI only
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - defensive fallback only
    linear_sum_assignment = None

ThresholdMethod = Literal["otsu", "min"]


@dataclass(frozen=True)
class InternalTrack2pCloneResult:
    """Result of the internal Track2p-style diagnostic matcher."""

    suite2p_indices: np.ndarray
    session_names: tuple[str, ...]
    thresholds: tuple[float, ...]
    match_counts: tuple[int, ...]


def run_internal_track2p_clone(
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    transform_type: str = "affine",
    iscell_threshold: float = 0.5,
    iou_dist_threshold: float = 16.0,
    threshold_method: ThresholdMethod = "otsu",
    threshold_remove_zeros: bool = False,
) -> InternalTrack2pCloneResult:
    """Run Track2p's default association logic inside BayesCaTrack.

    The implementation follows Track2p's default path closely:

    * load Suite2p ROIs using only ``iscell[:, 1] > iscell_threshold``;
    * keep overlap pixels and use binary masks;
    * register only consecutive session pairs;
    * build ``1 - IoU`` costs with a centroid-distance gate;
    * solve a Hungarian assignment for each consecutive pair;
    * threshold assigned-pair IoUs edge-locally; and
    * propagate tracks from first-session ROIs until the first missing link.
    """

    sessions = load_internal_track2p_clone_sessions(
        subject_dir,
        plane_name=plane_name,
        input_format=input_format,
        include_behavior=include_behavior,
        iscell_threshold=iscell_threshold,
    )
    if not sessions:
        raise ValueError(f"No Track2p-style sessions found under {subject_dir}")

    session_names = tuple(session.session_name for session in sessions)
    if len(sessions) == 1:
        roi_indices = _roi_indices_for_plane(sessions[0].plane_data)
        return InternalTrack2pCloneResult(
            suite2p_indices=roi_indices.reshape(-1, 1).astype(int),
            session_names=session_names,
            thresholds=(),
            match_counts=(),
        )

    consecutive_matches: list[dict[int, int]] = []
    thresholds: list[float] = []
    match_counts: list[int] = []
    for reference_session, moving_session in zip(sessions[:-1], sessions[1:], strict=True):
        registered_moving_plane = register_plane_pair(
            reference_session.plane_data,
            moving_session.plane_data,
            transform_type=transform_type,
        )
        match_mapping, threshold = _match_registered_pair_track2p_style(
            reference_session.plane_data,
            registered_moving_plane,
            iou_dist_threshold=iou_dist_threshold,
            threshold_method=threshold_method,
            threshold_remove_zeros=threshold_remove_zeros,
        )
        consecutive_matches.append(match_mapping)
        thresholds.append(float(threshold))
        match_counts.append(len(match_mapping))

    track_rows = build_track_rows_from_matches(
        session_names,
        consecutive_matches,
        start_roi_indices=None,
        start_session_index=0,
        fill_value=-1,
    )
    return InternalTrack2pCloneResult(
        suite2p_indices=track_rows,
        session_names=session_names,
        thresholds=tuple(thresholds),
        match_counts=tuple(match_counts),
    )


def load_internal_track2p_clone_sessions(
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    iscell_threshold: float = 0.5,
) -> list[Track2pSession]:
    """Load sessions using Track2p's ROI filtering semantics.

    This differs deliberately from :func:`load_track2p_subject` for Suite2p
    folders: Track2p's default loader filters only by the probability column of
    ``iscell.npy`` and does not remove Suite2p overlap pixels.
    """

    if input_format not in {"auto", "suite2p", "npy"}:
        raise ValueError("input_format must be 'auto', 'suite2p', or 'npy'")
    if not 0.0 <= iscell_threshold <= 1.0:
        raise ValueError("iscell_threshold must be between 0 and 1")

    subject_path = Path(subject_dir)
    sessions: list[Track2pSession] = []
    for session_dir in find_track2p_session_dirs(subject_path):
        suite2p_plane_dir = session_dir / "suite2p" / plane_name
        npy_plane_dir = session_dir / "data_npy" / plane_name

        plane_data: CalciumPlaneData | None = None
        if input_format in {"auto", "suite2p"} and suite2p_plane_dir.exists():
            plane_data = _load_suite2p_plane_track2p_style(
                suite2p_plane_dir,
                iscell_threshold=iscell_threshold,
            )
        elif input_format in {"auto", "npy"} and npy_plane_dir.exists():
            plane_data = load_raw_npy_plane(npy_plane_dir)

        if plane_data is None:
            if input_format == "auto":
                continue
            raise FileNotFoundError(
                f"Could not find {input_format} data for session {session_dir.name!r} and plane {plane_name!r}"
            )

        motion_energy = None
        if include_behavior:
            motion_energy_path = session_dir / "move_deve" / "motion_energy_glob.npy"
            if motion_energy_path.exists():
                motion_energy = np.load(motion_energy_path)

        sessions.append(
            Track2pSession(
                session_dir=session_dir,
                session_name=session_dir.name,
                session_date=_session_date_from_name(session_dir.name),
                plane_data=plane_data,
                motion_energy=motion_energy,
            )
        )
    return sessions


def _load_suite2p_plane_track2p_style(
    plane_dir: str | Path,
    *,
    iscell_threshold: float,
) -> CalciumPlaneData:
    plane_dir = Path(plane_dir)
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    if stat.ndim != 1:
        raise ValueError("Suite2p stat.npy must be a one-dimensional object array")

    iscell_path = plane_dir / "iscell.npy"
    iscell = np.load(iscell_path, allow_pickle=True) if iscell_path.exists() else None

    ops_path = plane_dir / "ops.npy"
    ops: dict[str, object] | None = None
    fov = None
    if ops_path.exists():
        loaded_ops = np.load(ops_path, allow_pickle=True).item()
        ops = dict(loaded_ops) if isinstance(loaded_ops, Mapping) else loaded_ops
        mean_image = loaded_ops.get("meanImg") if isinstance(loaded_ops, Mapping) else None
        if mean_image is not None:
            fov = np.asarray(mean_image)

    image_shape = _infer_suite2p_image_shape(stat, ops, fov)
    selected_indices: list[int] = []
    roi_masks: list[np.ndarray] = []
    cell_probabilities: list[float] = []

    for roi_index, roi_stat in enumerate(stat):
        probability = _track2p_iscell_probability(iscell, roi_index)
        if not _track2p_keep_roi(probability, iscell_threshold=iscell_threshold):
            continue

        ypix = np.asarray(roi_stat["ypix"], dtype=int)
        xpix = np.asarray(roi_stat["xpix"], dtype=int)
        if ypix.size == 0:
            continue

        mask = np.zeros(image_shape, dtype=bool)
        mask[ypix, xpix] = True
        selected_indices.append(int(roi_index))
        roi_masks.append(mask)
        cell_probabilities.append(float(probability))

    roi_mask_array = _stack_masks_or_empty(roi_masks, image_shape)
    if fov is None:
        fov = np.asarray(roi_mask_array, dtype=float).sum(axis=0)

    return CalciumPlaneData(
        roi_masks=roi_mask_array,
        fov=np.asarray(fov),
        cell_probabilities=np.asarray(cell_probabilities, dtype=float),
        roi_indices=np.asarray(selected_indices, dtype=int),
        source="suite2p_track2p_clone",
        plane_name=plane_dir.name,
        ops=ops,
    )


def _match_registered_pair_track2p_style(
    reference_plane: CalciumPlaneData,
    registered_moving_plane: CalciumPlaneData,
    *,
    iou_dist_threshold: float,
    threshold_method: ThresholdMethod,
    threshold_remove_zeros: bool,
) -> tuple[dict[int, int], float]:
    if linear_sum_assignment is None:
        raise ImportError("Internal Track2p clone requires scipy.optimize.linear_sum_assignment")
    if iou_dist_threshold < 0.0:
        raise ValueError("iou_dist_threshold must be non-negative")

    iou_matrix = _cross_iou_matrix(
        reference_plane.roi_masks,
        registered_moving_plane.roi_masks,
        dist_threshold=float(iou_dist_threshold),
    )
    if iou_matrix.shape[0] == 0 or iou_matrix.shape[1] == 0:
        return {}, float("inf")

    reference_positions, moving_positions = linear_sum_assignment(1.0 - iou_matrix)
    matched_ious = iou_matrix[reference_positions, moving_positions]
    threshold_values = matched_ious[matched_ious > 0.0] if threshold_remove_zeros else matched_ious
    threshold = _threshold_metric(threshold_values, method=threshold_method)
    keep = matched_ious > threshold

    reference_roi_indices = _roi_indices_for_plane(reference_plane)
    moving_roi_indices = _roi_indices_for_plane(registered_moving_plane)
    return (
        {
            int(reference_roi_indices[reference_position]): int(moving_roi_indices[moving_position])
            for reference_position, moving_position in zip(
                reference_positions[keep],
                moving_positions[keep],
                strict=True,
            )
        },
        float(threshold),
    )


def _cross_iou_matrix(
    reference_masks: np.ndarray,
    moving_masks: np.ndarray,
    *,
    dist_threshold: float,
) -> np.ndarray:
    reference_masks = np.asarray(reference_masks) > 0
    moving_masks = np.asarray(moving_masks) > 0
    if reference_masks.ndim != 3 or moving_masks.ndim != 3:
        raise ValueError("ROI masks must have shape (n_roi, height, width)")
    if reference_masks.shape[1:] != moving_masks.shape[1:]:
        raise ValueError("registered ROI masks must share the reference image shape")

    distances = _pairwise_centroid_distances(reference_masks, moving_masks)
    ious = np.zeros((reference_masks.shape[0], moving_masks.shape[0]), dtype=float)
    for reference_index in range(reference_masks.shape[0]):
        reference_mask = reference_masks[reference_index]
        for moving_index in range(moving_masks.shape[0]):
            if distances[reference_index, moving_index] > dist_threshold:
                continue
            moving_mask = moving_masks[moving_index]
            union = np.logical_or(reference_mask, moving_mask)
            union_size = int(np.sum(union))
            if union_size == 0:
                continue
            intersection_size = int(np.sum(np.logical_and(reference_mask, moving_mask)))
            ious[reference_index, moving_index] = intersection_size / union_size
    return ious


def _pairwise_centroid_distances(reference_masks: np.ndarray, moving_masks: np.ndarray) -> np.ndarray:
    reference_centroids = _mask_centroids(reference_masks)
    moving_centroids = _mask_centroids(moving_masks)
    diffs = reference_centroids[:, None, :] - moving_centroids[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def _mask_centroids(masks: np.ndarray) -> np.ndarray:
    centroids = np.zeros((masks.shape[0], 2), dtype=float)
    for roi_index, mask in enumerate(masks):
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            continue
        centroids[roi_index] = np.array([float(np.mean(rows)), float(np.mean(cols))])
    return centroids


def _threshold_metric(values: np.ndarray, *, method: ThresholdMethod) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    if method == "otsu":
        try:
            from skimage.filters import threshold_otsu  # type: ignore[import-not-found]

            return float(threshold_otsu(values))
        except ImportError:
            return _threshold_otsu_numpy(values)
    if method == "min":
        try:
            from skimage.filters import threshold_minimum  # type: ignore[import-not-found]

            return float(threshold_minimum(values))
        except ImportError:
            return _threshold_minimum_numpy(values)
    raise ValueError("threshold_method must be 'otsu' or 'min'")


def _threshold_otsu_numpy(values: np.ndarray) -> float:
    if values.size == 1 or float(np.min(values)) == float(np.max(values)):
        return float(values[0])
    counts, bin_edges = np.histogram(values, bins=256)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight_left = np.cumsum(counts, dtype=float)
    weight_right = np.cumsum(counts[::-1], dtype=float)[::-1]
    nonzero_left = weight_left > 0
    nonzero_right = weight_right > 0
    mean_left = np.divide(
        np.cumsum(counts * centers, dtype=float),
        weight_left,
        out=np.zeros_like(weight_left, dtype=float),
        where=nonzero_left,
    )
    mean_right = np.divide(
        np.cumsum((counts * centers)[::-1], dtype=float)[::-1],
        weight_right,
        out=np.zeros_like(weight_right, dtype=float),
        where=nonzero_right,
    )
    variances = weight_left[:-1] * weight_right[1:] * (mean_left[:-1] - mean_right[1:]) ** 2
    if variances.size == 0:
        return float(values[0])
    return float(centers[:-1][int(np.argmax(variances))])


def _threshold_minimum_numpy(values: np.ndarray) -> float:
    if values.size == 1 or float(np.min(values)) == float(np.max(values)):
        return float(values[0])
    counts, bin_edges = np.histogram(values, bins=256)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    minima = np.flatnonzero((counts[1:-1] <= counts[:-2]) & (counts[1:-1] <= counts[2:])) + 1
    if minima.size == 0:
        return _threshold_otsu_numpy(values)
    return float(centers[int(minima[0])])


def _track2p_iscell_probability(iscell: np.ndarray | None, roi_index: int) -> float:
    if iscell is None:
        return 1.0
    if np.ndim(iscell) == 2 and iscell.shape[1] > 1:
        return float(iscell[roi_index, 1])
    return float(np.asarray(iscell)[roi_index])


def _track2p_keep_roi(probability: float, *, iscell_threshold: float) -> bool:
    return bool(probability > iscell_threshold)


def _infer_suite2p_image_shape(
    stat: np.ndarray,
    ops: Mapping[str, object] | None,
    fov: np.ndarray | None,
) -> tuple[int, int]:
    if ops is not None and "Ly" in ops and "Lx" in ops:
        return int(ops["Ly"]), int(ops["Lx"])
    if fov is not None:
        if fov.ndim != 2:
            raise ValueError("Suite2p meanImg must be two-dimensional")
        return int(fov.shape[0]), int(fov.shape[1])
    max_y = -1
    max_x = -1
    for roi_stat in stat:
        ypix = np.asarray(roi_stat["ypix"], dtype=int)
        xpix = np.asarray(roi_stat["xpix"], dtype=int)
        if ypix.size:
            max_y = max(max_y, int(np.max(ypix)))
            max_x = max(max_x, int(np.max(xpix)))
    if max_y < 0 or max_x < 0:
        raise ValueError("Cannot infer image shape from empty Suite2p stat.npy without ops.npy")
    return max_y + 1, max_x + 1


def _stack_masks_or_empty(masks: list[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros((0, *image_shape), dtype=bool)
    return np.stack(masks, axis=0).astype(bool, copy=False)


def _roi_indices_for_plane(plane: CalciumPlaneData) -> np.ndarray:
    if plane.roi_indices is not None:
        return np.asarray(plane.roi_indices, dtype=int)
    return np.arange(plane.n_rois, dtype=int)


def _session_date_from_name(session_name: str) -> date | None:
    try:
        return date.fromisoformat(session_name.split("_", maxsplit=1)[0])
    except ValueError:
        return None
