"""Delta audit for gap-rescue edges added after component cleanup.

The promoted Track2p-policy component cleanup row is the baseline.  Gap rescue
is useful only if a later method can admit a tiny number of high-confidence
missing links without inheriting the broad false-positive set from the global
gap-rescue row.  This audit compares that baseline against a gap-rescue
candidate and exports only the edges that are newly introduced by gap rescue.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_emulation_benchmark import (
    ThresholdMethod,
    emulate_track2p_tracks,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
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
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyPruneConfig,
    _margin_against_competitor,
    _roi_indices,
    _threshold_assigned_iou,
    _track2p_cross_iou_diagnostic_matrices,
    emulate_track2p_pruned_tracks,
)
from bayescatrack.track2p_registration import register_plane_pair
from scipy.optimize import linear_sum_assignment

GapAuditEdge = tuple[int, int, int, int]


@dataclass(frozen=True)
class GapEdgeFeature:
    """Registration and local-assignment features for one accepted edge."""

    registered_iou: float
    threshold: float
    threshold_margin: float
    centroid_distance: float
    area_ratio: float
    row_rank: int
    column_rank: int
    row_margin: float
    column_margin: float


@dataclass(frozen=True)
class CandidateEdgeContext:
    """Track-level context for one candidate edge occurrence."""

    track_id: int
    seed_roi: int
    source_chain_length: int
    target_chain_length: int
    upstream_support: int
    downstream_support: int
    triplet_support: int
    component_status_after: str
    creates_complete_track_fp: int


@dataclass(frozen=True)
class GapEdgeAuditResult:
    """Edge delta audit rows plus compact per-subject summary rows."""

    edge_rows: tuple[dict[str, float | int | str], ...]
    summary_rows: tuple[dict[str, float | int | str], ...]


def run_track2p_policy_gap_edge_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int = 2,
    cleanup_config: ComponentCleanupConfig | None = None,
    base_prediction: str = "component-cleanup",
    candidate_prediction: str = "gap-component-no-apply",
) -> GapEdgeAuditResult:
    """Return newly introduced gap-rescue edges absent from component cleanup."""

    if base_prediction != "component-cleanup":
        raise ValueError("Only base-prediction='component-cleanup' is supported")
    if candidate_prediction != "gap-component-no-apply":
        raise ValueError(
            "Only candidate-prediction='gap-component-no-apply' is supported"
        )
    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
        max_gap=max_gap,
    )
    if int(policy_config.max_gap) < 1:
        raise ValueError("max_gap must be at least 1")

    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    cleanup_config = cleanup_config or ComponentCleanupConfig()
    all_edge_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy gap-edge audit requires independent manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )

        base_full = _component_cleanup_prediction(
            sessions,
            reference_tracks,
            config=policy_config,
            cleanup_config=cleanup_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
        )
        candidate_full = _normalize_int_track_matrix(
            emulate_track2p_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                max_gap=int(policy_config.max_gap),
            )
        )
        base_eval, reference_eval, _ = _evaluated_prediction_rows(
            base_full,
            reference_tracks,
            config=policy_config,
        )
        candidate_eval, _, _ = _evaluated_prediction_rows(
            candidate_full,
            reference_tracks,
            config=policy_config,
        )
        feature_index = _edge_feature_index(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            max_gap=int(policy_config.max_gap),
        )
        consensus_feature_indexes = tuple(
            _edge_feature_index(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(threshold),
                max_gap=int(policy_config.max_gap),
            )
            for threshold in _consensus_thresholds(float(iou_distance_threshold))
        )
        subject_rows = _delta_edge_rows(
            subject=subject_dir.name,
            sessions=sessions,
            base_tracks=base_eval,
            candidate_tracks=candidate_eval,
            reference_tracks=reference_eval,
            feature_index=feature_index,
            consensus_feature_indexes=consensus_feature_indexes,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            transform_type=policy_config.transform_type,
            cell_probability_threshold=float(policy_config.cell_probability_threshold),
            max_gap=int(policy_config.max_gap),
        )
        all_edge_rows.extend(subject_rows)
        summary_rows.append(
            _summary_row(
                subject_dir.name,
                subject_rows,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                max_gap=int(policy_config.max_gap),
            )
        )

    summary_rows.append(
        _summary_row(
            "ALL",
            all_edge_rows,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            max_gap=int(policy_config.max_gap),
        )
    )
    return GapEdgeAuditResult(tuple(all_edge_rows), tuple(summary_rows))


def _component_cleanup_prediction(
    sessions: Sequence[Track2pSession],
    reference_tracks: np.ndarray,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> np.ndarray:
    prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    predicted_full = _normalize_int_track_matrix(prediction.tracks)
    predicted_eval, reference_eval, evaluated_track_ids = _evaluated_prediction_rows(
        predicted_full,
        reference_tracks,
        config=config,
    )
    audit_rows = component_audit_rows(
        predicted_eval,
        reference_eval,
        sessions=sessions,
        diagnostics=prediction.diagnostics,
        subject="",
        config=cleanup_config,
        track_ids=evaluated_track_ids,
        seed_session=config.seed_session,
    )
    return apply_weakest_bridge_splits(
        predicted_full,
        _mark_applied_splits(audit_rows, apply_splits=True),
    )


def _delta_edge_rows(
    *,
    subject: str,
    sessions: Sequence[Track2pSession],
    base_tracks: np.ndarray,
    candidate_tracks: np.ndarray,
    reference_tracks: np.ndarray,
    feature_index: Mapping[GapAuditEdge, GapEdgeFeature],
    consensus_feature_indexes: Sequence[Mapping[GapAuditEdge, GapEdgeFeature]],
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    transform_type: str,
    cell_probability_threshold: float,
    max_gap: int,
) -> list[dict[str, float | int | str]]:
    base_counts = _observed_neighbor_edge_counter(base_tracks, max_gap=max_gap)
    candidate_counts = _observed_neighbor_edge_counter(
        candidate_tracks, max_gap=max_gap
    )
    reference_counts = _all_forward_edge_counter(reference_tracks, max_gap=max_gap)
    base_component_status = _component_status_by_seed(base_tracks, reference_tracks)
    candidate_contexts = _candidate_edge_contexts(
        candidate_tracks,
        reference_tracks,
        max_gap=max_gap,
    )

    rows: list[dict[str, float | int | str]] = []
    for edge in sorted(candidate_counts):
        delta_count = int(candidate_counts[edge] - base_counts.get(edge, 0))
        if delta_count <= 0:
            continue
        session_a, session_b, roi_a, roi_b = edge
        contexts = candidate_contexts.get(edge, ())
        for occurrence_index in range(delta_count):
            context = (
                contexts[min(occurrence_index, len(contexts) - 1)]
                if contexts
                else CandidateEdgeContext(
                    track_id=-1,
                    seed_roi=-1,
                    source_chain_length=0,
                    target_chain_length=0,
                    upstream_support=0,
                    downstream_support=0,
                    triplet_support=0,
                    component_status_after="unknown",
                    creates_complete_track_fp=0,
                )
            )
            feature = feature_index.get(edge)
            rows.append(
                {
                    "subject": subject,
                    "session_a": int(session_a),
                    "session_b": int(session_b),
                    "session_a_name": str(sessions[session_a].session_name),
                    "session_b_name": str(sessions[session_b].session_name),
                    "roi_a": int(roi_a),
                    "roi_b": int(roi_b),
                    "edge_status_against_gt": (
                        "true_positive"
                        if reference_counts.get(edge, 0) > 0
                        else "false_positive"
                    ),
                    "gap_length": int(session_b - session_a),
                    "registered_iou": _feature_value(feature, "registered_iou"),
                    "shifted_iou": float("nan"),
                    "roi_aware_shifted_score": float("nan"),
                    "centroid_distance": _feature_value(feature, "centroid_distance"),
                    "area_ratio": _feature_value(feature, "area_ratio"),
                    "row_rank": _feature_int(feature, "row_rank"),
                    "column_rank": _feature_int(feature, "column_rank"),
                    "row_margin": _feature_value(feature, "row_margin"),
                    "column_margin": _feature_value(feature, "column_margin"),
                    "threshold": _feature_value(feature, "threshold"),
                    "threshold_margin": _feature_value(feature, "threshold_margin"),
                    "cell_probability_a": _cell_probability(
                        sessions[session_a], int(roi_a)
                    ),
                    "cell_probability_b": _cell_probability(
                        sessions[session_b], int(roi_b)
                    ),
                    "source_chain_length": int(context.source_chain_length),
                    "target_chain_length": int(context.target_chain_length),
                    "upstream_support": int(context.upstream_support),
                    "downstream_support": int(context.downstream_support),
                    "triplet_support": int(context.triplet_support),
                    "consensus_votes": int(
                        sum(1 for index in consensus_feature_indexes if edge in index)
                    ),
                    "component_status_before": str(
                        base_component_status.get(context.seed_roi, "absent")
                    ),
                    "component_status_after": context.component_status_after,
                    "creates_complete_track_fp": int(context.creates_complete_track_fp),
                    "candidate_track_id": int(context.track_id),
                    "candidate_seed_roi": int(context.seed_roi),
                    "delta_count": int(delta_count),
                    "threshold_method": str(threshold_method),
                    "iou_distance_threshold": float(iou_distance_threshold),
                    "cell_probability_threshold": float(cell_probability_threshold),
                    "transform_type": str(transform_type),
                    "max_gap": int(max_gap),
                }
            )
    return rows


def _observed_neighbor_edge_counter(
    track_matrix: np.ndarray, *, max_gap: int
) -> Counter[GapAuditEdge]:
    counter: Counter[GapAuditEdge] = Counter()
    for row in _normalize_int_track_matrix(track_matrix):
        for edge in _observed_neighbor_edges(row, max_gap=max_gap):
            counter[edge] += 1
    return counter


def _candidate_edge_contexts(
    track_matrix: np.ndarray,
    reference_tracks: np.ndarray,
    *,
    max_gap: int,
) -> dict[GapAuditEdge, tuple[CandidateEdgeContext, ...]]:
    reference_complete = _complete_track_counter(reference_tracks)
    contexts: dict[GapAuditEdge, list[CandidateEdgeContext]] = defaultdict(list)
    for track_id, row in enumerate(_normalize_int_track_matrix(track_matrix)):
        valid_sessions = tuple(int(index) for index in np.flatnonzero(row >= 0))
        seed_roi = int(row[0]) if row.size and row[0] >= 0 else -1
        component_status = _complete_status(row, reference_complete)
        creates_complete_fp = int(component_status == "false_positive")
        for session_a, session_b, roi_a, roi_b in _observed_neighbor_edges(
            row, max_gap=max_gap
        ):
            source_chain_length = int(
                sum(session <= session_a for session in valid_sessions)
            )
            target_chain_length = int(
                sum(session >= session_b for session in valid_sessions)
            )
            upstream_support = max(0, source_chain_length - 1)
            downstream_support = max(0, target_chain_length - 1)
            contexts[(session_a, session_b, roi_a, roi_b)].append(
                CandidateEdgeContext(
                    track_id=int(track_id),
                    seed_roi=seed_roi,
                    source_chain_length=source_chain_length,
                    target_chain_length=target_chain_length,
                    upstream_support=upstream_support,
                    downstream_support=downstream_support,
                    triplet_support=int(
                        upstream_support > 0 and downstream_support > 0
                    ),
                    component_status_after=component_status,
                    creates_complete_track_fp=creates_complete_fp,
                )
            )
    return {key: tuple(value) for key, value in contexts.items()}


def _observed_neighbor_edges(
    row: np.ndarray, *, max_gap: int
) -> Iterable[GapAuditEdge]:
    valid_sessions = tuple(int(index) for index in np.flatnonzero(row >= 0))
    for left, right in zip(valid_sessions, valid_sessions[1:], strict=False):
        if right - left > int(max_gap):
            continue
        yield (left, right, int(row[left]), int(row[right]))


def _all_forward_edge_counter(
    track_matrix: np.ndarray, *, max_gap: int
) -> Counter[GapAuditEdge]:
    counter: Counter[GapAuditEdge] = Counter()
    for row in _normalize_int_track_matrix(track_matrix):
        valid_sessions = tuple(int(index) for index in np.flatnonzero(row >= 0))
        for left_index, session_a in enumerate(valid_sessions):
            for session_b in valid_sessions[left_index + 1 :]:
                if session_b - session_a > int(max_gap):
                    break
                counter[
                    (session_a, session_b, int(row[session_a]), int(row[session_b]))
                ] += 1
    return counter


def _component_status_by_seed(
    track_matrix: np.ndarray, reference_tracks: np.ndarray
) -> dict[int, str]:
    reference_complete = _complete_track_counter(reference_tracks)
    output: dict[int, str] = {}
    for row in _normalize_int_track_matrix(track_matrix):
        if row.size == 0 or row[0] < 0:
            continue
        output.setdefault(int(row[0]), _complete_status(row, reference_complete))
    return output


def _complete_track_counter(track_matrix: np.ndarray) -> Counter[tuple[int, ...]]:
    counter: Counter[tuple[int, ...]] = Counter()
    for row in _normalize_int_track_matrix(track_matrix):
        if np.all(row >= 0):
            counter[tuple(int(value) for value in row)] += 1
    return counter


def _complete_status(
    row: np.ndarray, reference_complete: Counter[tuple[int, ...]]
) -> str:
    if not np.all(row >= 0):
        return "incomplete"
    return (
        "true_positive"
        if reference_complete.get(tuple(int(value) for value in row), 0) > 0
        else "false_positive"
    )


def _edge_feature_index(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    max_gap: int,
) -> dict[GapAuditEdge, GapEdgeFeature]:
    sessions = tuple(sessions)
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    output: dict[GapAuditEdge, GapEdgeFeature] = {}
    for session_a in range(max(0, len(sessions) - 1)):
        max_step = min(int(max_gap), len(sessions) - 1 - session_a)
        for step in range(1, max_step + 1):
            session_b = session_a + step
            pair_features = _accepted_pair_features(
                sessions[session_a],
                sessions[session_b],
                transform_type=transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold) * float(step),
            )
            source_indices = roi_indices_by_session[session_a]
            target_indices = roi_indices_by_session[session_b]
            for (local_a, local_b), feature in pair_features.items():
                output[
                    (
                        session_a,
                        session_b,
                        int(source_indices[local_a]),
                        int(target_indices[local_b]),
                    )
                ] = feature
    return output


def _accepted_pair_features(
    reference_session: Track2pSession,
    moving_session: Track2pSession,
    *,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> dict[tuple[int, int], GapEdgeFeature]:
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
    if iou.size == 0:
        return {}
    row_ind, col_ind = linear_sum_assignment(1.0 - iou)
    assigned_iou = iou[row_ind, col_ind]
    threshold = _threshold_assigned_iou(assigned_iou, method=threshold_method)
    output: dict[tuple[int, int], GapEdgeFeature] = {}
    for row, column, value in zip(row_ind, col_ind, assigned_iou, strict=True):
        if float(value) <= float(threshold):
            continue
        output[(int(row), int(column))] = GapEdgeFeature(
            registered_iou=float(value),
            threshold=float(threshold),
            threshold_margin=float(value) - float(threshold),
            centroid_distance=float(distances[row, column]),
            area_ratio=float(area_ratios[row, column]),
            row_rank=_rank_descending(iou[row, :], selected_index=int(column)),
            column_rank=_rank_descending(iou[:, column], selected_index=int(row)),
            row_margin=_margin_against_competitor(
                iou[row, :], selected_index=int(column)
            ),
            column_margin=_margin_against_competitor(
                iou[:, column], selected_index=int(row)
            ),
        )
    return output


def _rank_descending(values: np.ndarray, *, selected_index: int) -> int:
    values = np.asarray(values, dtype=float).reshape(-1)
    selected = float(values[int(selected_index)])
    return int(1 + np.sum(values > selected))


def _cell_probability(session: Track2pSession, suite2p_roi: int) -> float:
    probabilities = session.plane_data.cell_probabilities
    if probabilities is None:
        return float("nan")
    roi_indices = _roi_indices(session)
    matches = np.flatnonzero(roi_indices == int(suite2p_roi))
    if matches.size == 0:
        return float("nan")
    return float(np.asarray(probabilities, dtype=float)[int(matches[0])])


def _feature_value(feature: GapEdgeFeature | None, name: str) -> float:
    if feature is None:
        return float("nan")
    return float(getattr(feature, name))


def _feature_int(feature: GapEdgeFeature | None, name: str) -> int:
    if feature is None:
        return -1
    return int(getattr(feature, name))


def _consensus_thresholds(iou_distance_threshold: float) -> tuple[float, float, float]:
    center = float(iou_distance_threshold)
    return (max(0.0, center - 2.0), center, center + 2.0)


def _summary_row(
    subject: str,
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    max_gap: int,
) -> dict[str, float | int | str]:
    true_positive = sum(
        1 for row in rows if row.get("edge_status_against_gt") == "true_positive"
    )
    false_positive = sum(
        1 for row in rows if row.get("edge_status_against_gt") == "false_positive"
    )
    return {
        "subject": subject,
        "delta_edges": int(len(rows)),
        "true_positive_edges": int(true_positive),
        "false_positive_edges": int(false_positive),
        "consecutive_edges": int(
            sum(int(row.get("gap_length", 0)) == 1 for row in rows)
        ),
        "skip_edges": int(sum(int(row.get("gap_length", 0)) > 1 for row in rows)),
        "complete_track_fp_edges": int(
            sum(int(row.get("creates_complete_track_fp", 0)) for row in rows)
        ),
        "threshold_method": str(threshold_method),
        "iou_distance_threshold": float(iou_distance_threshold),
        "max_gap": int(max_gap),
    }


def write_gap_edge_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write gap-edge audit rows as CSV or JSON."""

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
    """Build the command-line parser for the gap-edge delta audit."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-gap-edge-audit",
        description=(
            "Audit only gap-rescue candidate edges that are absent from the "
            "Track2p-policy component-cleanup baseline."
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
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--base-prediction",
        choices=("component-cleanup",),
        default="component-cleanup",
    )
    parser.add_argument(
        "--candidate-prediction",
        choices=("gap-component-no-apply",),
        default="gap-component-no-apply",
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
    """Run the Track2p-policy gap-edge audit CLI."""

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
        max_gap=args.max_gap,
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
    result = run_track2p_policy_gap_edge_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        max_gap=int(args.max_gap),
        cleanup_config=cleanup_config,
        base_prediction=args.base_prediction,
        candidate_prediction=args.candidate_prediction,
    )
    write_gap_edge_rows(result.edge_rows, args.output, output_format=args.format)
    if args.summary_output is not None:
        write_gap_edge_rows(
            result.summary_rows,
            args.summary_output,
            output_format=args.format,
        )
    return 0


def _no_prune_config() -> Track2pPolicyPruneConfig:
    return Track2pPolicyPruneConfig(
        threshold_margin=0.0,
        competition_margin=0.0,
        min_area_ratio=0.0,
        centroid_distance=float("inf"),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
