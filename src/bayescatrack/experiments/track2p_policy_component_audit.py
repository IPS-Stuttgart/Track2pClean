"""Component-level audit and weakest-bridge cleanup for Track2p-policy rows.

The edge-level prune diagnostics showed that local edge features do not cleanly
separate false-positive from true-positive links.  This module therefore audits
whole predicted policy tracks and can apply one conservative operation per
component: split the component at its weakest bridge when the component-level
risk score is high enough.  By default, cleanup is restricted to complete
predicted tracks so the post-processing targets complete-track precision without
unnecessarily damaging partial tracks used by pairwise scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.evaluation.track2p_metrics import normalize_track_matrix
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _filter_tracks_by_seed_rois,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _reference_seed_roi_set,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_policy_audit import track_edge_counter
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
    Track2pPolicyPruneConfig,
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_COMPONENT_CLEANUP_METHOD = "track2p-policy-component-cleanup"


@dataclass(frozen=True)
class ComponentCleanupConfig:
    """Unsupervised weakest-bridge split parameters."""

    threshold_margin_scale: float = 0.10
    competition_margin_scale: float = 0.20
    area_ratio_floor: float = 0.45
    centroid_distance_scale: float = 4.0
    split_risk_threshold: float = 1.50
    split_penalty: float = 0.25
    min_side_observations: int = 2
    require_complete_track: bool = True
    threshold_margin_weight: float = 1.0
    row_margin_weight: float = 1.0
    column_margin_weight: float = 1.0
    centroid_distance_weight: float = 1.0
    area_ratio_weight: float = 1.0

    def __post_init__(self) -> None:
        _require_positive(self.threshold_margin_scale, name="threshold_margin_scale")
        _require_positive(
            self.competition_margin_scale, name="competition_margin_scale"
        )
        _require_positive(self.area_ratio_floor, name="area_ratio_floor")
        _require_positive(self.centroid_distance_scale, name="centroid_distance_scale")
        _require_nonnegative(self.split_risk_threshold, name="split_risk_threshold")
        _require_nonnegative(self.split_penalty, name="split_penalty")
        object.__setattr__(
            self,
            "min_side_observations",
            _positive_int_value(
                self.min_side_observations, name="min_side_observations"
            ),
        )


@dataclass(frozen=True)
class ComponentAuditOutput:
    """Benchmark rows plus component diagnostics."""

    results: tuple[SubjectBenchmarkResult, ...]
    component_rows: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True)
class _ComponentEdge:
    session_index: int
    source_roi: int
    target_roi: int
    diagnostic: Track2pPolicyLinkDiagnostic | None
    risk: float


def run_track2p_policy_component_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    apply_splits: bool = True,
) -> ComponentAuditOutput:
    """Run component-level policy audit and optional weakest-bridge cleanup."""

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
    results: list[SubjectBenchmarkResult] = []
    component_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy component audit requires independent manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=_no_prune_config(),
        )
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        predicted_full = _normalize_int_track_matrix(prediction.tracks)
        predicted_eval, reference_eval, evaluated_track_ids = (
            _evaluated_prediction_rows(
                predicted_full,
                reference_tracks,
                config=policy_config,
            )
        )
        audit_rows = component_audit_rows(
            predicted_eval,
            reference_eval,
            sessions=sessions,
            diagnostics=prediction.diagnostics,
            subject=subject_dir.name,
            config=cleanup_config,
            track_ids=evaluated_track_ids,
            seed_session=policy_config.seed_session,
        )
        subject_rows = _mark_applied_splits(
            audit_rows,
            apply_splits=apply_splits,
        )
        cleaned = (
            apply_weakest_bridge_splits(
                predicted_full,
                subject_rows,
            )
            if apply_splits
            else predicted_full
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        applied_splits = int(
            sum(int(row["applied_split"]) for row in subject_rows)
            if apply_splits
            else 0
        )
        candidate_splits = int(
            sum(int(row["would_split_at_weakest_edge"]) for row in subject_rows)
        )
        scores = {
            **scores,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_component_apply_splits": int(apply_splits),
            "track2p_component_candidate_splits": candidate_splits,
            "track2p_component_applied_splits": applied_splits,
            "track2p_component_split_risk_threshold": float(
                cleanup_config.split_risk_threshold
            ),
            "track2p_component_split_penalty": float(cleanup_config.split_penalty),
            "track2p_component_min_side_observations": int(
                cleanup_config.min_side_observations
            ),
            "track2p_component_require_complete_track": int(
                cleanup_config.require_complete_track
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=(
                    "Track2p-policy weakest-bridge component split"
                    if apply_splits
                    else "Track2p-policy component audit"
                ),
                method=cast(Any, TRACK2P_POLICY_COMPONENT_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        component_rows.extend(
            _with_metadata(
                subject_rows,
                {
                    "threshold_method": str(threshold_method),
                    "iou_distance_threshold": float(iou_distance_threshold),
                    "cell_probability_threshold": float(
                        policy_config.cell_probability_threshold
                    ),
                    "transform_type": str(policy_config.transform_type),
                },
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(component_rows))


def component_audit_rows(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    sessions: Sequence[Track2pSession],
    diagnostics: Sequence[Track2pPolicyLinkDiagnostic],
    subject: str = "",
    config: ComponentCleanupConfig | None = None,
    track_ids: Sequence[int] | None = None,
    seed_session: int = 0,
) -> list[dict[str, float | int | str]]:
    """Return one audit row per predicted policy component."""

    config = config or ComponentCleanupConfig()
    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    reference = _normalize_int_track_matrix(reference_track_matrix)
    seed_session = int(seed_session)
    if seed_session < 0 or seed_session >= predicted.shape[1]:
        raise IndexError(
            f"seed_session {seed_session} out of bounds for {predicted.shape[1]} sessions"
        )
    ids = (
        tuple(range(predicted.shape[0]))
        if track_ids is None
        else tuple(int(track_id) for track_id in track_ids)
    )
    if len(ids) != predicted.shape[0]:
        raise ValueError("track_ids must have one entry per predicted track")
    diagnostic_by_edge = _diagnostics_by_suite2p_edge(sessions, diagnostics)
    predicted_edge_counts = track_edge_counter(predicted)
    reference_edge_counts = track_edge_counter(reference)
    observation_counts = _observation_counter(predicted)
    reference_by_seed = _reference_rows_by_seed(reference, seed_session=seed_session)
    reference_complete_counts = _complete_track_counts(reference)
    rows: list[dict[str, float | int | str]] = []
    for row_index, track in enumerate(predicted):
        component_id = ids[row_index]
        edges = _component_edges(track, diagnostic_by_edge, config=config)
        weakest = max(edges, key=lambda edge: edge.risk, default=None)
        valid_observations = int(np.sum(track >= 0))
        is_complete_track = bool(valid_observations == int(predicted.shape[1]))
        split_index = -1 if weakest is None else int(weakest.session_index)
        left_observations, right_observations = _split_observation_counts(
            track, split_index
        )
        split_gain = (
            _split_gain(
                weakest.risk if weakest is not None else 0.0,
                left_observations=left_observations,
                right_observations=right_observations,
                config=config,
            )
            if weakest is not None
            else -float(config.split_penalty)
        )
        would_split = bool(
            weakest is not None
            and weakest.risk >= config.split_risk_threshold
            and left_observations >= config.min_side_observations
            and right_observations >= config.min_side_observations
            and (is_complete_track or not config.require_complete_track)
            and split_gain > 0.0
        )
        pairwise_tp, pairwise_fp = _component_pairwise_counts(
            edges,
            predicted_edge_counts=predicted_edge_counts,
            reference_edge_counts=reference_edge_counts,
        )
        rows.append(
            {
                "subject": subject,
                "predicted_track_id": int(component_id),
                "n_sessions": valid_observations,
                "total_sessions": int(predicted.shape[1]),
                "n_rois": valid_observations,
                "n_edges": int(len(edges)),
                "is_complete_track": int(is_complete_track),
                "min_edge_margin": _min_or_nan(
                    _diagnostic_value(edge, "threshold_margin") for edge in edges
                ),
                "median_edge_margin": _median_or_nan(
                    _diagnostic_value(edge, "threshold_margin") for edge in edges
                ),
                "n_marginal_edges": int(
                    sum(_is_marginal_edge(edge, config=config) for edge in edges)
                ),
                "max_centroid_distance": _max_or_nan(
                    _diagnostic_value(edge, "centroid_distance") for edge in edges
                ),
                "min_area_ratio": _min_or_nan(
                    _diagnostic_value(edge, "area_ratio") for edge in edges
                ),
                "n_boundary_edges": int(
                    sum(_is_boundary_edge(edge, config=config) for edge in edges)
                ),
                "n_same_predicted_edges": int(
                    sum(
                        predicted_edge_counts[
                            (
                                edge.session_index,
                                edge.session_index + 1,
                                edge.source_roi,
                                edge.target_roi,
                            )
                        ]
                        > 1
                        for edge in edges
                    )
                ),
                "n_conflicting_edges": int(
                    sum(
                        observation_counts[(edge.session_index, edge.source_roi)] > 1
                        or observation_counts[(edge.session_index + 1, edge.target_roi)]
                        > 1
                        for edge in edges
                    )
                ),
                "component_score": float(weakest.risk if weakest else 0.0),
                "component_risk_sum": float(sum(edge.risk for edge in edges)),
                "weakest_bridge_session_a": split_index,
                "weakest_bridge_session_b": split_index + 1 if split_index >= 0 else -1,
                "weakest_bridge_source_roi": int(weakest.source_roi) if weakest else -1,
                "weakest_bridge_target_roi": int(weakest.target_roi) if weakest else -1,
                "weakest_bridge_risk": float(weakest.risk if weakest else 0.0),
                "split_gain": float(split_gain),
                "would_split_at_weakest_edge": int(would_split),
                "applied_split": 0,
                "complete_track_status_against_gt": _complete_track_status(
                    track, reference_complete_counts
                ),
                "pairwise_tp_edges": int(pairwise_tp),
                "pairwise_fp_edges": int(pairwise_fp),
                "pairwise_fn_edges": int(
                    _component_pairwise_false_negatives(
                        track,
                        reference_by_seed,
                        seed_session=seed_session,
                    )
                ),
            }
        )
    return rows


def apply_weakest_bridge_splits(
    predicted_track_matrix: Any,
    component_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Split components marked by ``component_audit_rows`` at their weakest bridge."""

    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    output: list[np.ndarray] = []
    rows_by_component = {int(row["predicted_track_id"]): row for row in component_rows}
    for component_id, track in enumerate(predicted):
        row = rows_by_component.get(component_id)
        if row is None or int(row.get("applied_split", 0)) == 0:
            output.append(np.asarray(track, dtype=int).copy())
            continue
        split_index = int(row["weakest_bridge_session_a"])
        left, right = split_track_at_bridge(track, split_index)
        output.append(left)
        output.append(right)
    if not output:
        return predicted[:0]
    return np.vstack(output).astype(int, copy=False)


def _evaluated_prediction_rows(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    config: Track2pBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    predicted = _normalize_int_track_matrix(predicted)
    reference = _normalize_int_track_matrix(reference)
    if not config.restrict_to_reference_seed_rois:
        return predicted, reference, tuple(range(predicted.shape[0]))
    reference_seed_rois = _reference_seed_roi_set(
        reference, seed_session=config.seed_session
    )
    keep_indices = tuple(
        index
        for index, row in enumerate(predicted)
        if _valid_seed_roi(row, reference_seed_rois, seed_session=config.seed_session)
    )
    predicted_eval = predicted[np.asarray(keep_indices, dtype=int)]
    reference_eval = _filter_tracks_by_seed_rois(
        reference,
        reference_seed_rois,
        seed_session=config.seed_session,
    )
    return predicted_eval, reference_eval, keep_indices


def _mark_applied_splits(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    apply_splits: bool,
) -> list[dict[str, float | int | str]]:
    marked: list[dict[str, float | int | str]] = []
    for row in rows:
        updated = dict(row)
        updated["applied_split"] = (
            int(row.get("would_split_at_weakest_edge", 0)) if apply_splits else 0
        )
        marked.append(updated)
    return marked


def split_track_at_bridge(
    track: Any, session_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right fragments after removing the bridge after ``session_index``."""

    row = _track_row_as_int(track)
    if session_index < 0 or session_index >= row.size - 1:
        raise IndexError("session_index must identify a consecutive bridge")
    left = row.copy()
    right = row.copy()
    left[session_index + 1 :] = -1
    right[: session_index + 1] = -1
    return left, right


def edge_risk_score(
    diagnostic: Track2pPolicyLinkDiagnostic | None,
    *,
    config: ComponentCleanupConfig | None = None,
) -> float:
    """Return an unsupervised risk score for one policy bridge."""

    if diagnostic is None:
        return 0.0
    config = config or ComponentCleanupConfig()
    return float(
        config.threshold_margin_weight
        * _low_value_risk(
            diagnostic.threshold_margin, scale=config.threshold_margin_scale
        )
        + config.row_margin_weight
        * _low_value_risk(diagnostic.row_margin, scale=config.competition_margin_scale)
        + config.column_margin_weight
        * _low_value_risk(
            diagnostic.column_margin, scale=config.competition_margin_scale
        )
        + config.centroid_distance_weight
        * _high_value_risk(
            diagnostic.centroid_distance, scale=config.centroid_distance_scale
        )
        + config.area_ratio_weight
        * _low_value_risk(diagnostic.area_ratio, scale=config.area_ratio_floor)
    )


def write_component_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write component diagnostics as CSV or JSON."""

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
    """Build the command-line parser for component-level policy cleanup."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-component-audit",
        description="Audit Track2p-policy components and optionally split weakest bridges.",
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
    parser.add_argument(
        "--apply-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply weakest-bridge splits before scoring; use --no-apply-splits for audit-only output.",
    )
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only split components observed in every session; use "
            "--no-require-complete-track to allow incomplete component splits."
        ),
    )
    parser.add_argument("--threshold-margin-scale", type=float, default=0.10)
    parser.add_argument("--competition-margin-scale", type=float, default=0.20)
    parser.add_argument("--area-ratio-floor", type=float, default=0.45)
    parser.add_argument("--centroid-distance-scale", type=float, default=4.0)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
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
    parser.add_argument("--component-output", type=Path, default=None)
    parser.add_argument("--component-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy component audit CLI."""

    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        threshold_margin_scale=args.threshold_margin_scale,
        competition_margin_scale=args.competition_margin_scale,
        area_ratio_floor=args.area_ratio_floor,
        centroid_distance_scale=args.centroid_distance_scale,
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
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    output = run_track2p_policy_component_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
        cleanup_config=cleanup_config,
        apply_splits=args.apply_splits,
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.component_output is not None:
        write_component_rows(
            output.component_rows,
            args.component_output,
            output_format=cast(Literal["csv", "json"], args.component_format),
        )
    return 0


def _diagnostics_by_suite2p_edge(
    sessions: Sequence[Track2pSession],
    diagnostics: Sequence[Track2pPolicyLinkDiagnostic],
) -> dict[tuple[int, int, int], Track2pPolicyLinkDiagnostic]:
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    output: dict[tuple[int, int, int], Track2pPolicyLinkDiagnostic] = {}
    for diagnostic in diagnostics:
        session_index = int(diagnostic.session_index)
        source_roi = int(roi_indices_by_session[session_index][diagnostic.local_roi_a])
        target_roi = int(
            roi_indices_by_session[session_index + 1][diagnostic.local_roi_b]
        )
        output[(session_index, source_roi, target_roi)] = diagnostic
    return output


def _component_edges(
    track: np.ndarray,
    diagnostic_by_edge: Mapping[tuple[int, int, int], Track2pPolicyLinkDiagnostic],
    *,
    config: ComponentCleanupConfig,
) -> list[_ComponentEdge]:
    edges: list[_ComponentEdge] = []
    for session_index in range(max(0, track.size - 1)):
        source = track[session_index]
        target = track[session_index + 1]
        if source < 0 or target < 0:
            continue
        diagnostic = diagnostic_by_edge.get((session_index, int(source), int(target)))
        edges.append(
            _ComponentEdge(
                session_index=session_index,
                source_roi=int(source),
                target_roi=int(target),
                diagnostic=diagnostic,
                risk=edge_risk_score(diagnostic, config=config),
            )
        )
    return edges


def _component_pairwise_counts(
    edges: Sequence[_ComponentEdge],
    *,
    predicted_edge_counts: Counter[tuple[int, int, int, int]],
    reference_edge_counts: Counter[tuple[int, int, int, int]],
) -> tuple[int, int]:
    true_positive = 0
    false_positive = 0
    for edge in edges:
        key = (
            edge.session_index,
            edge.session_index + 1,
            edge.source_roi,
            edge.target_roi,
        )
        if reference_edge_counts.get(key, 0) > 0:
            true_positive += 1
        elif predicted_edge_counts.get(key, 0) > 0:
            false_positive += 1
    return true_positive, false_positive


def _component_pairwise_false_negatives(
    track: np.ndarray,
    reference_by_seed: Mapping[int, np.ndarray],
    *,
    seed_session: int = 0,
) -> int:
    seed = (
        int(track[seed_session])
        if 0 <= seed_session < track.size and track[seed_session] >= 0
        else -1
    )
    reference = reference_by_seed.get(seed)
    if reference is None:
        return 0
    predicted_edges = {
        (index, int(track[index]), int(track[index + 1]))
        for index in range(track.size - 1)
        if track[index] >= 0 and track[index + 1] >= 0
    }
    reference_edges = {
        (index, int(reference[index]), int(reference[index + 1]))
        for index in range(reference.size - 1)
        if reference[index] >= 0 and reference[index + 1] >= 0
    }
    return len(reference_edges - predicted_edges)


def _reference_rows_by_seed(
    reference: np.ndarray, *, seed_session: int = 0
) -> dict[int, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    for row in reference:
        if 0 <= seed_session < row.size and row[seed_session] >= 0:
            rows.setdefault(int(row[seed_session]), row)
    return rows


def _complete_track_counts(track_matrix: np.ndarray) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for row in track_matrix:
        if np.all(row >= 0):
            counts[tuple(int(value) for value in row)] += 1
    return counts


def _complete_track_status(
    track: np.ndarray,
    reference_complete_counts: Counter[tuple[int, ...]],
) -> str:
    if not np.all(track >= 0):
        return "incomplete"
    key = tuple(int(value) for value in track)
    return (
        "true_positive"
        if reference_complete_counts.get(key, 0) > 0
        else "false_positive"
    )


def _observation_counter(track_matrix: np.ndarray) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for row in track_matrix:
        for session_index, roi in enumerate(row):
            if roi >= 0:
                counts[(session_index, int(roi))] += 1
    return counts


def _normalize_int_track_matrix(track_matrix: Any) -> np.ndarray:
    matrix = normalize_track_matrix(track_matrix)
    output = np.full(matrix.shape, -1, dtype=int)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if _valid_roi(value):
                output[row_index, column_index] = int(value)
    return output


def _track_row_as_int(track: Any) -> np.ndarray:
    row = np.asarray(track, dtype=object).reshape(-1)
    output = np.full(row.shape, -1, dtype=int)
    for index, value in enumerate(row):
        if _valid_roi(value):
            output[index] = int(value)
    return output


def _valid_roi(value: Any) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric >= 0.0)


def _valid_seed_roi(
    row: np.ndarray,
    reference_seed_rois: set[int],
    *,
    seed_session: int,
) -> bool:
    if seed_session < 0 or seed_session >= row.size:
        return False
    value = row[seed_session]
    return bool(_valid_roi(value) and int(value) in reference_seed_rois)


def _split_observation_counts(track: np.ndarray, session_index: int) -> tuple[int, int]:
    if session_index < 0:
        return int(np.sum(track >= 0)), 0
    return (
        int(np.sum(track[: session_index + 1] >= 0)),
        int(np.sum(track[session_index + 1 :] >= 0)),
    )


def _split_gain(
    risk: float,
    *,
    left_observations: int,
    right_observations: int,
    config: ComponentCleanupConfig,
) -> float:
    short_side_penalty = 0.0
    if left_observations < config.min_side_observations:
        short_side_penalty += config.split_penalty
    if right_observations < config.min_side_observations:
        short_side_penalty += config.split_penalty
    return float(risk - config.split_penalty - short_side_penalty)


def _diagnostic_value(edge: _ComponentEdge, name: str) -> float:
    diagnostic = edge.diagnostic
    if diagnostic is None:
        return float("nan")
    return float(getattr(diagnostic, name))


def _is_marginal_edge(edge: _ComponentEdge, *, config: ComponentCleanupConfig) -> bool:
    diagnostic = edge.diagnostic
    if diagnostic is None:
        return False
    return bool(
        diagnostic.threshold_margin <= config.threshold_margin_scale
        and (
            diagnostic.row_margin <= config.competition_margin_scale
            or diagnostic.column_margin <= config.competition_margin_scale
        )
    )


def _is_boundary_edge(edge: _ComponentEdge, *, config: ComponentCleanupConfig) -> bool:
    diagnostic = edge.diagnostic
    if diagnostic is None:
        return False
    return bool(
        diagnostic.centroid_distance >= config.centroid_distance_scale
        or diagnostic.area_ratio < config.area_ratio_floor
    )


def _low_value_risk(value: float, *, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, (float(scale) - float(value)) / float(scale)))


def _high_value_risk(value: float, *, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, float(value) / float(scale) - 1.0))


def _min_or_nan(values: Sequence[float] | Any) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(min(finite)) if finite else float("nan")


def _max_or_nan(values: Sequence[float] | Any) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else float("nan")


def _median_or_nan(values: Sequence[float] | Any) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def _with_metadata(
    rows: Sequence[Mapping[str, float | int | str]],
    metadata: Mapping[str, Any],
) -> list[dict[str, float | int | str]]:
    formatted = {key: _format_metadata_value(value) for key, value in metadata.items()}
    return [{**dict(row), **formatted} for row in rows]


def _roi_indices(session: Track2pSession) -> np.ndarray:
    roi_indices = session.plane_data.roi_indices
    if roi_indices is None:
        return np.arange(session.plane_data.n_rois, dtype=int)
    return np.asarray(roi_indices, dtype=int)


def _no_prune_config() -> Track2pPolicyPruneConfig:
    return Track2pPolicyPruneConfig(
        threshold_margin=0.0,
        competition_margin=0.0,
        min_area_ratio=0.0,
        centroid_distance=float("inf"),
    )


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _require_positive(value: float, *, name: str) -> None:
    if _finite_float_value(value, name=name) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(value: float, *, name: str) -> None:
    if _finite_float_value(value, name=name) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _finite_float_value(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _positive_int_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
    else:
        raise ValueError(f"{name} must be a positive integer")
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
