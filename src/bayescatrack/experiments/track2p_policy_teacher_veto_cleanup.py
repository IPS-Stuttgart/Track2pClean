"""Track2p-teacher veto cleanup after component cleanup.

The residual audit after ComponentCleanup shows a small remaining pairwise-FP
budget.  Track2p still has the best pairwise F1, so its *absence* can be useful
negative teacher evidence.  This candidate starts from the frozen
Track2pPolicy component-cleanup prediction and conservatively splits only weak,
Bayes-only adjacent edges that are absent from Track2p output.

The operation is intentionally narrow and disabled from any paper-facing
manifest by default.  It is a candidate cleanup row to test whether Track2p can
veto low-support false continuations without giving back ComponentCleanup's
complete-track gain.  No manual-GT labels are used to select vetoes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
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
from bayescatrack.experiments.track2p_policy_audit import TrackEdge, track_edge_counter
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentAuditOutput,
    ComponentCleanupConfig,
    _evaluated_prediction_rows,
    _mark_applied_splits,
    _normalize_int_track_matrix,
    apply_weakest_bridge_splits,
    component_audit_rows,
)
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    ResidualFeature,
    _feature_index_from_policy_diagnostics,
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_TEACHER_VETO_CLEANUP_METHOD = "track2p-policy-teacher-veto-cleanup"
TeacherVetoEdgeOrder = Literal["lexicographic", "risk"]
TeacherVetoOrder = Literal["lexicographic", "weakest"]


@dataclass(frozen=True)
class TeacherVetoConfig:
    """Gate for conservative Track2p-teacher edge vetoes."""

    max_threshold_margin: float = 0.10
    max_competition_margin: float = 0.20
    min_registered_iou: float | None = None
    max_registered_iou: float | None = None
    min_centroid_distance: float | None = None
    max_area_ratio: float | None = None
    max_cell_probability: float | None = None
    require_unassigned_by_hungarian: bool = False
    require_teacher_conflict: bool = False
    allow_complete_track_veto: bool = False
    complete_track_veto_only: bool = False
    keep_right_fragment: bool = True
    min_fragment_observations: int = 2
    edge_order: TeacherVetoEdgeOrder = "risk"
    veto_order: TeacherVetoOrder | None = None
    max_applied_vetoes: int | None = None


@dataclass(frozen=True)
class TeacherVetoReport:
    """Prediction plus per-edge veto diagnostics."""

    tracks: np.ndarray
    rows: tuple[dict[str, float | int | str], ...]


def run_track2p_policy_teacher_veto_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    veto_config: TeacherVetoConfig | None = None,
) -> ComponentAuditOutput:
    """Run component cleanup followed by conservative Track2p-teacher vetoes."""

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

    cleanup_config = cleanup_config or ComponentCleanupConfig()
    veto_config = veto_config or TeacherVetoConfig()
    results: list[SubjectBenchmarkResult] = []
    veto_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy teacher-veto cleanup requires independent manual GT "
                "references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        base_full, features = _component_cleanup_prediction_with_features(
            sessions,
            reference_tracks,
            config=policy_config,
            cleanup_config=cleanup_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            subject=subject_dir.name,
        )
        teacher_full, _variant = _predict_subject_tracks(
            subject_dir, replace(policy_config, method="track2p-baseline")
        )
        veto = apply_teacher_veto_edges(
            base_full,
            teacher_full,
            feature_index=features,
            config=veto_config,
        )
        scores = _score_prediction_against_reference(
            veto.tracks, reference, config=policy_config
        )
        applied = int(sum(int(row["applied"]) for row in veto.rows))
        candidates = int(len(veto.rows))
        scores = {
            **scores,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_teacher_veto_candidates": candidates,
            "track2p_teacher_veto_applied": applied,
            "track2p_teacher_veto_max_threshold_margin": float(
                veto_config.max_threshold_margin
            ),
            "track2p_teacher_veto_max_competition_margin": float(
                veto_config.max_competition_margin
            ),
            "track2p_teacher_veto_min_registered_iou": _optional_float(
                veto_config.min_registered_iou
            ),
            "track2p_teacher_veto_max_registered_iou": _optional_float(
                veto_config.max_registered_iou
            ),
            "track2p_teacher_veto_min_centroid_distance": _optional_float(
                veto_config.min_centroid_distance
            ),
            "track2p_teacher_veto_max_area_ratio": _optional_float(
                veto_config.max_area_ratio
            ),
            "track2p_teacher_veto_max_cell_probability": _optional_float(
                veto_config.max_cell_probability
            ),
            "track2p_teacher_veto_require_unassigned_by_hungarian": int(
                veto_config.require_unassigned_by_hungarian
            ),
            "track2p_teacher_veto_require_teacher_conflict": int(
                veto_config.require_teacher_conflict
            ),
            "track2p_teacher_veto_allow_complete_track_veto": int(
                veto_config.allow_complete_track_veto
            ),
            "track2p_teacher_veto_complete_track_veto_only": int(
                veto_config.complete_track_veto_only
            ),
            "track2p_teacher_veto_keep_right_fragment": int(
                veto_config.keep_right_fragment
            ),
            "track2p_teacher_veto_min_fragment_observations": int(
                veto_config.min_fragment_observations
            ),
            "track2p_teacher_veto_edge_order": _resolved_edge_order(veto_config),
            "track2p_teacher_veto_order": _legacy_veto_order_label(veto_config),
            "track2p_teacher_veto_max_applied_vetoes": (
                -1
                if veto_config.max_applied_vetoes is None
                else int(veto_config.max_applied_vetoes)
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant="Track2p-policy component cleanup + teacher veto cleanup",
                method=cast(Any, TRACK2P_POLICY_TEACHER_VETO_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        veto_rows.extend(
            {
                **row,
                "subject": subject_dir.name,
                "threshold_method": str(threshold_method),
                "iou_distance_threshold": f"{float(iou_distance_threshold):g}",
                "cell_probability_threshold": (
                    f"{float(policy_config.cell_probability_threshold):g}"
                ),
                "transform_type": str(policy_config.transform_type),
            }
            for row in veto.rows
        )
    return ComponentAuditOutput(tuple(results), tuple(veto_rows))


def _component_cleanup_prediction_with_features(
    sessions: Sequence[Track2pSession],
    reference_tracks: np.ndarray,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    subject: str,
) -> tuple[np.ndarray, dict[TrackEdge, ResidualFeature]]:
    prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    policy_full = _normalize_int_track_matrix(prediction.tracks)
    policy_eval, reference_eval, evaluated_track_ids = _evaluated_prediction_rows(
        policy_full, reference_tracks, config=config
    )
    audit_rows = component_audit_rows(
        policy_eval,
        reference_eval,
        sessions=sessions,
        diagnostics=prediction.diagnostics,
        subject=subject,
        config=cleanup_config,
        track_ids=evaluated_track_ids,
        seed_session=config.seed_session,
    )
    cleaned_full = apply_weakest_bridge_splits(
        policy_full, _mark_applied_splits(audit_rows, apply_splits=True)
    )
    return cleaned_full, _feature_index_from_policy_diagnostics(
        sessions, prediction.diagnostics
    )


def apply_teacher_veto_edges(
    predicted_track_matrix: Any,
    teacher_track_matrix: Any,
    *,
    feature_index: Mapping[TrackEdge, ResidualFeature] | None = None,
    config: TeacherVetoConfig | None = None,
) -> TeacherVetoReport:
    """Split weak adjacent predicted edges that Track2p does not support."""

    output = _normalize_int_track_matrix(predicted_track_matrix)
    teacher = _normalize_int_track_matrix(teacher_track_matrix)
    features = dict(feature_index or {})
    config = config or TeacherVetoConfig()
    rows: list[dict[str, float | int | str]] = []

    teacher_counts = track_edge_counter(teacher)
    applied_count = 0
    for edge in _ordered_veto_edges(output, teacher, features, config):
        while track_edge_counter(output).get(edge, 0) > teacher_counts.get(edge, 0):
            if _max_applied_vetoes_reached(applied_count, config.max_applied_vetoes):
                rows.append(
                    _veto_limit_row(edge, features.get(edge, ResidualFeature()))
                )
                return TeacherVetoReport(output, tuple(rows))
            output, row = _try_veto_edge(
                output,
                edge,
                teacher=teacher,
                feature=features.get(edge),
                config=config,
            )
            rows.append(row)
            if int(row["applied"]) == 0:
                break
            applied_count += 1
    return TeacherVetoReport(output, tuple(rows))


def _ordered_veto_edges(
    predicted: np.ndarray,
    teacher: np.ndarray,
    feature_index: Mapping[TrackEdge, ResidualFeature],
    config: TeacherVetoConfig,
) -> tuple[TrackEdge, ...]:
    teacher_counts = track_edge_counter(teacher)
    candidate_edges = tuple(
        edge
        for edge, predicted_count in track_edge_counter(predicted).items()
        if int(predicted_count) > int(teacher_counts.get(edge, 0))
    )
    edge_order = _resolved_edge_order(config)
    if edge_order == "lexicographic":
        return tuple(sorted(candidate_edges))
    if edge_order != "risk":
        raise ValueError(f"Unsupported teacher-veto edge order: {edge_order!r}")
    return tuple(
        sorted(
            candidate_edges,
            key=lambda edge: _veto_edge_risk_order_key(edge, feature_index.get(edge)),
        )
    )


def _resolved_edge_order(config: TeacherVetoConfig) -> TeacherVetoEdgeOrder:
    if config.veto_order == "weakest":
        return "risk"
    if config.veto_order == "lexicographic":
        return "lexicographic"
    return config.edge_order


def _legacy_veto_order_label(config: TeacherVetoConfig) -> str:
    if config.veto_order is not None:
        return str(config.veto_order)
    return "weakest" if _resolved_edge_order(config) == "risk" else "lexicographic"


def _veto_edge_risk_order_key(
    edge: TrackEdge, feature: ResidualFeature | None
) -> tuple[int, float, float, float, float, float, float, int, int, int, int]:
    feature = feature or ResidualFeature()
    competition_margin = min(float(feature.row_margin), float(feature.column_margin))
    min_cell_probability = _min_cell_probability(feature)
    feature_missing = int(
        not _finite(feature.threshold_margin)
        or not _finite(competition_margin)
        or not _finite(feature.registered_iou)
        or not _finite(feature.centroid_distance)
        or not _finite(feature.area_ratio)
        or not _finite(min_cell_probability)
    )
    return (
        feature_missing,
        _finite_or_inf(feature.threshold_margin),
        _finite_or_inf(competition_margin),
        _finite_or_inf(feature.registered_iou),
        _finite_or_neg_inf(-float(feature.centroid_distance)),
        _finite_or_inf(feature.area_ratio),
        _finite_or_inf(min_cell_probability),
        int(edge[0]),
        int(edge[1]),
        int(edge[2]),
        int(edge[3]),
    )


def _max_applied_vetoes_reached(
    applied_count: int, max_applied_vetoes: int | None
) -> bool:
    return max_applied_vetoes is not None and int(applied_count) >= int(
        max(0, int(max_applied_vetoes))
    )


def _veto_limit_row(
    edge: TrackEdge, feature: ResidualFeature
) -> dict[str, float | int | str]:
    row = _veto_row(edge, feature)
    row["reason"] = "max_applied_vetoes_reached"
    return row


def _try_veto_edge(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    teacher: np.ndarray,
    feature: ResidualFeature | None,
    config: TeacherVetoConfig,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    output = np.asarray(predicted, dtype=int).copy()
    session_a, session_b, roi_a, roi_b = edge
    feature = feature or ResidualFeature()
    row = _veto_row(edge, feature)
    teacher_conflict = _has_teacher_conflict(edge, teacher)
    row["teacher_conflict"] = int(teacher_conflict)
    candidate_rows = tuple(
        int(row_index)
        for row_index in np.flatnonzero(
            (output[:, session_a] == roi_a) & (output[:, session_b] == roi_b)
        )
    )
    if session_b != session_a + 1:
        row["reason"] = "not_adjacent"
        return output, row
    if not candidate_rows:
        row["reason"] = "edge_not_found"
        return output, row
    if len(candidate_rows) > 1:
        row["reason"] = "ambiguous_predicted_edge"
        return output, row
    row_index = int(candidate_rows[0])
    row["row_index"] = row_index
    row_is_complete = bool(np.all(output[row_index] >= 0))
    if config.complete_track_veto_only and not row_is_complete:
        row["reason"] = "not_complete_track"
        return output, row
    if row_is_complete and not config.allow_complete_track_veto:
        row["reason"] = "would_split_complete_track"
        return output, row
    if config.require_teacher_conflict and not teacher_conflict:
        row["reason"] = "no_teacher_conflict"
        return output, row
    reason = _gate_reject_reason(feature, config)
    if reason is not None:
        row["reason"] = reason
        return output, row

    left = output[row_index].copy()
    right = output[row_index].copy()
    left[session_b:] = -1
    right[:session_b] = -1
    left_observations = int(np.sum(left >= 0))
    right_observations = int(np.sum(right >= 0))
    min_fragment_observations = max(1, int(config.min_fragment_observations))
    if (
        left_observations < min_fragment_observations
        or right_observations < min_fragment_observations
    ):
        row["reason"] = "fragment_too_short"
        row["left_observations"] = left_observations
        row["right_observations"] = right_observations
        return output, row

    output[row_index] = left
    if config.keep_right_fragment and np.any(right >= 0):
        output = np.vstack([output, right.reshape(1, -1)])
    row["applied"] = 1
    row["reason"] = "accepted_split_edge"
    row["left_observations"] = left_observations
    row["right_observations"] = right_observations
    return output, row


def _has_teacher_conflict(edge: TrackEdge, teacher: np.ndarray) -> bool:
    """Return whether Track2p actively prefers another adjacent endpoint."""

    session_a, session_b, roi_a, roi_b = edge
    if session_a < 0 or session_b < 0 or teacher.ndim != 2:
        return False
    if session_a >= teacher.shape[1] or session_b >= teacher.shape[1]:
        return False
    for row in teacher:
        teacher_source = int(row[session_a])
        teacher_target = int(row[session_b])
        source_conflict = (
            teacher_source == int(roi_a)
            and teacher_target >= 0
            and teacher_target != int(roi_b)
        )
        target_conflict = (
            teacher_target == int(roi_b)
            and teacher_source >= 0
            and teacher_source != int(roi_a)
        )
        if source_conflict or target_conflict:
            return True
    return False


def _gate_reject_reason(
    feature: ResidualFeature, config: TeacherVetoConfig
) -> str | None:
    if not _finite(feature.threshold_margin):
        return "feature_missing"
    if float(feature.threshold_margin) > config.max_threshold_margin:
        return "threshold_margin_too_high"
    competition_margin = min(float(feature.row_margin), float(feature.column_margin))
    if not _finite(competition_margin):
        return "feature_missing"
    if competition_margin > config.max_competition_margin:
        return "competition_margin_too_high"
    if config.min_registered_iou is not None:
        if not _finite(feature.registered_iou):
            return "feature_missing"
        if float(feature.registered_iou) < float(config.min_registered_iou):
            return "registered_iou_too_low"
    if config.max_registered_iou is not None:
        if not _finite(feature.registered_iou):
            return "feature_missing"
        if float(feature.registered_iou) > float(config.max_registered_iou):
            return "registered_iou_too_high"
    if config.min_centroid_distance is not None:
        if not _finite(feature.centroid_distance):
            return "feature_missing"
        if float(feature.centroid_distance) < float(config.min_centroid_distance):
            return "centroid_distance_too_low"
    if config.max_area_ratio is not None:
        if not _finite(feature.area_ratio):
            return "feature_missing"
        if float(feature.area_ratio) > float(config.max_area_ratio):
            return "area_ratio_too_high"
    if config.max_cell_probability is not None:
        min_cell_probability = _min_cell_probability(feature)
        if not _finite(min_cell_probability):
            return "feature_missing"
        if float(min_cell_probability) > float(config.max_cell_probability):
            return "cell_probability_too_high"
    if config.require_unassigned_by_hungarian and int(feature.assigned_by_hungarian):
        return "assigned_by_hungarian"
    return None


def _veto_row(
    edge: TrackEdge, feature: ResidualFeature
) -> dict[str, float | int | str]:
    session_a, session_b, roi_a, roi_b = edge
    return {
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "row_index": -1,
        "applied": 0,
        "reason": "not_evaluated",
        "registered_iou": float(feature.registered_iou),
        "centroid_distance": float(feature.centroid_distance),
        "area_ratio": float(feature.area_ratio),
        "cell_probability_a": float(feature.cell_probability_a),
        "cell_probability_b": float(feature.cell_probability_b),
        "min_cell_probability": float(_min_cell_probability(feature)),
        "row_rank": int(feature.row_rank),
        "column_rank": int(feature.column_rank),
        "row_margin": float(feature.row_margin),
        "column_margin": float(feature.column_margin),
        "threshold": float(feature.threshold),
        "threshold_margin": float(feature.threshold_margin),
        "assigned_by_hungarian": int(feature.assigned_by_hungarian),
        "teacher_conflict": -1,
        "left_observations": 0,
        "right_observations": 0,
    }


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _min_cell_probability(feature: ResidualFeature) -> float:
    return min(float(feature.cell_probability_a), float(feature.cell_probability_b))


def _finite_or_neg_inf(value: float) -> float:
    if _finite(value):
        return float(value)
    return float("inf")


def _finite_or_inf(value: float) -> float:
    return float(value) if _finite(value) else float("inf")


def _optional_float(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def write_veto_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
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
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-teacher-veto-cleanup",
        description=(
            "Run Track2pPolicy component cleanup and then split weak Bayes-only "
            "adjacent edges that Track2p does not support."
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
    parser.add_argument("--max-threshold-margin", type=float, default=0.10)
    parser.add_argument("--max-competition-margin", type=float, default=0.20)
    parser.add_argument("--min-registered-iou", type=float, default=None)
    parser.add_argument("--max-registered-iou", type=float, default=None)
    parser.add_argument(
        "--min-centroid-distance",
        type=float,
        default=None,
        help="Only veto teacher-absent edges whose centroid distance is at least this large.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=None,
        help="Only veto teacher-absent edges whose ROI area ratio is at most this value.",
    )
    parser.add_argument(
        "--max-cell-probability",
        type=float,
        default=None,
        help=(
            "Only veto teacher-absent edges whose weaker endpoint cell probability "
            "is at most this value."
        ),
    )
    parser.add_argument(
        "--require-unassigned-by-hungarian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only veto teacher-absent edges that were not assigned by the local Hungarian step.",
    )
    parser.add_argument(
        "--require-teacher-conflict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only veto Bayes-only edges when Track2p actively chooses a competing "
            "source or target in the same adjacent session pair."
        ),
    )
    parser.add_argument(
        "--allow-complete-track-veto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow vetoes that split currently complete predicted rows. Disabled "
            "by default to protect ComponentCleanup's complete-track gain."
        ),
    )
    parser.add_argument(
        "--complete-track-veto-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only apply teacher-veto splits to currently complete predicted rows. "
            "This targets the residual complete-track FP budget without spending "
            "a small veto cap on incomplete-row pairwise FPs first."
        ),
    )
    parser.add_argument(
        "--keep-right-fragment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the target-side fragment after a veto split for auditability.",
    )
    parser.add_argument(
        "--min-veto-fragment-observations",
        type=int,
        default=2,
        help=(
            "Require both fragments created by a teacher-veto split to contain "
            "at least this many observations."
        ),
    )
    parser.add_argument(
        "--teacher-veto-edge-order",
        choices=("lexicographic", "risk"),
        default="risk",
        help=(
            "Order candidate Bayes-only veto edges. 'risk' tries the weakest "
            "low-margin edges first and is useful with --max-applied-vetoes."
        ),
    )
    parser.add_argument(
        "--veto-order",
        choices=("lexicographic", "weakest"),
        default=None,
        help=(
            "Compatibility alias for --teacher-veto-edge-order; 'weakest' maps "
            "to 'risk'."
        ),
    )
    parser.add_argument("--max-applied-vetoes", type=int, default=None)
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
    parser.add_argument("--diagnostics-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    veto_config = TeacherVetoConfig(
        max_threshold_margin=float(args.max_threshold_margin),
        max_competition_margin=float(args.max_competition_margin),
        min_registered_iou=args.min_registered_iou,
        max_registered_iou=args.max_registered_iou,
        min_centroid_distance=args.min_centroid_distance,
        max_area_ratio=args.max_area_ratio,
        max_cell_probability=args.max_cell_probability,
        require_unassigned_by_hungarian=bool(args.require_unassigned_by_hungarian),
        require_teacher_conflict=bool(args.require_teacher_conflict),
        allow_complete_track_veto=bool(args.allow_complete_track_veto),
        complete_track_veto_only=bool(args.complete_track_veto_only),
        keep_right_fragment=bool(args.keep_right_fragment),
        min_fragment_observations=int(args.min_veto_fragment_observations),
        edge_order=cast(TeacherVetoEdgeOrder, args.teacher_veto_edge_order),
        veto_order=cast(TeacherVetoOrder | None, args.veto_order),
        max_applied_vetoes=args.max_applied_vetoes,
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
        max_gap=1,
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
    output = run_track2p_policy_teacher_veto_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
        cleanup_config=cleanup_config,
        veto_config=veto_config,
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.diagnostics_output is not None:
        write_veto_rows(
            output.component_rows,
            args.diagnostics_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
