"""Coherence-gated suffix-stitch what-if audit.

This module starts from Track2pPolicy ComponentCleanup, generates the same
short suffix candidates as the suffix ranking audit, applies a hard
trajectory-coherence gate, and scores the result after at most one suffix
stitch per subject.  The command can be used as the promoted method row or as
an audit row when candidate diagnostics are requested.
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
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
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
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
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
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
    emulate_track2p_pruned_tracks,
)
from bayescatrack.experiments.track2p_policy_suffix_stitch_ranking_audit import (
    _FeatureCache,
    _PathCandidate,
    _max_attr,
    _mean_attr,
    _min_attr,
    _nanmin_default,
    _path_row,
    _ranked_suffix_paths,
)

TRACK2P_POLICY_COHERENCE_SUFFIX_STITCH_WHATIF_METHOD = "track2p-policy-coherence-suffix-stitch-whatif"
TRACK2P_POLICY_COHERENCE_SUFFIX_STITCH_METHOD = "track2p-policy-coherence-suffix-stitch"


@dataclass(frozen=True)
class CoherenceSuffixStitchGate:
    """Hard gate for exploratory suffix-stitch what-if candidates."""

    suffix_path_length: int = 2
    min_cell_probability: float = 0.80
    min_area_ratio: float = 0.80
    max_centroid_distance: float = 6.0
    min_shifted_iou: float = 0.30
    min_motion_consistency: float = 0.50
    min_shape_consistency: float = 0.82
    max_stitches_per_subject: int = 1


@dataclass(frozen=True)
class CoherenceSuffixStitchWhatIfResult:
    """Scored what-if rows and candidate audit rows."""

    result_rows: tuple[dict[str, float | int | str], ...]
    candidate_rows: tuple[dict[str, float | int | str], ...]


def run_track2p_policy_coherence_suffix_stitch_whatif(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    gate: CoherenceSuffixStitchGate | None = None,
    edge_top_k: int = 25,
    path_beam_width: int = 100,
) -> CoherenceSuffixStitchWhatIfResult:
    """Return coherence-gated suffix-stitch what-if rows."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(f"No Track2p-style subject directories found under {policy_config.data}")

    cleanup_config = cleanup_config or ComponentCleanupConfig()
    gate = gate or CoherenceSuffixStitchGate()
    result_rows: list[dict[str, float | int | str]] = []
    candidate_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        subject_result, subject_candidates = _subject_whatif_rows(
            subject_dir,
            config=policy_config,
            cleanup_config=cleanup_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            gate=gate,
            edge_top_k=int(edge_top_k),
            path_beam_width=int(path_beam_width),
        )
        result_rows.append(subject_result)
        candidate_rows.extend(subject_candidates)
    result_rows.append(_aggregate_result_row(result_rows, gate=gate))
    return CoherenceSuffixStitchWhatIfResult(tuple(result_rows), tuple(candidate_rows))


def _subject_whatif_rows(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    gate: CoherenceSuffixStitchGate,
    edge_top_k: int,
    path_beam_width: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    reference = _load_reference_for_subject(subject_dir, data_root=config.data, config=config)
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError("Track2p-policy coherence suffix-stitch what-if requires independent manual GT references")
    sessions = _load_subject_sessions(subject_dir, config)
    _validate_reference_roi_indices(reference, sessions)
    reference_tracks = _reference_matrix(reference, curated_only=config.curated_only)
    cleaned_eval, reference_eval = _component_cleanup_eval(
        sessions,
        reference_tracks,
        subject=subject_dir.name,
        config=config,
        cleanup_config=cleanup_config,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
    )
    feature_cache = _FeatureCache(
        sessions=sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(config.cell_probability_threshold),
        matrices={},
    )
    paths = _ranked_suffix_paths(
        cleaned_eval,
        reference_eval,
        subject=subject_dir.name,
        feature_cache=feature_cache,
        max_suffix_length=int(gate.suffix_path_length),
        edge_top_k=int(edge_top_k),
        path_beam_width=int(path_beam_width),
    )
    baseline_scores = dict(score_track_matrices(cleaned_eval, reference_eval))
    selected = _select_paths(
        paths,
        cleaned_eval,
        reference_eval,
        gate=gate,
    )
    stitched = _apply_suffix_paths(cleaned_eval, selected)
    stitched_scores = dict(score_track_matrices(stitched, reference_eval))
    candidate_rows = [
        _candidate_row(
            subject_dir.name,
            path,
            cleaned_eval,
            reference_eval,
            baseline_scores=baseline_scores,
            selected=path in selected,
            gate=gate,
        )
        for path in paths
    ]
    return (
        _result_row(
            subject_dir.name,
            baseline_scores,
            stitched_scores,
            selected,
            candidate_rows,
            gate=gate,
        ),
        candidate_rows,
    )


def _component_cleanup_eval(
    sessions: Sequence[Any],
    reference_tracks: np.ndarray,
    *,
    subject: str,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    policy_full = _normalize_int_track_matrix(prediction.tracks)
    policy_eval, reference_eval, evaluated_track_ids = _evaluated_prediction_rows(policy_full, reference_tracks, config=config)
    audit_rows = component_audit_rows(
        policy_eval,
        reference_eval,
        sessions=sessions,
        diagnostics=cast(Sequence[Track2pPolicyLinkDiagnostic], prediction.diagnostics),
        subject=subject,
        config=cleanup_config,
        track_ids=evaluated_track_ids,
        seed_session=config.seed_session,
    )
    cleaned_full = apply_weakest_bridge_splits(policy_full, _mark_applied_splits(audit_rows, apply_splits=True))
    cleaned_eval, reference_eval, _ = _evaluated_prediction_rows(cleaned_full, reference_tracks, config=config)
    return cleaned_eval, reference_eval


def _select_paths(
    paths: Sequence[_PathCandidate],
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    gate: CoherenceSuffixStitchGate,
) -> tuple[_PathCandidate, ...]:
    passing = [path for path in paths if _passes_coherence_gate(path, predicted, reference, gate=gate)]
    passing.sort(key=_coherence_sort_key)
    return tuple(passing[: int(gate.max_stitches_per_subject)])


def _passes_coherence_gate(
    path: _PathCandidate,
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    gate: CoherenceSuffixStitchGate,
) -> bool:
    metrics = _path_metrics(path)
    if len(path.edges) != int(gate.suffix_path_length):
        return False
    if not _would_reach_final_session(path, predicted):
        return False
    if any(_target_slot_occupied(path, predicted)):
        return False
    row = _path_row("", path, predicted, reference)
    if int(row["creates_duplicate_source"]) or int(row["creates_duplicate_target"]):
        return False
    if int(row["would_merge_complete_tp"]):
        return False
    return bool(
        metrics["min_cell_probability"] >= float(gate.min_cell_probability)
        and metrics["min_area_ratio"] >= float(gate.min_area_ratio)
        and metrics["max_centroid_distance"] <= float(gate.max_centroid_distance)
        and metrics["min_shifted_iou"] >= float(gate.min_shifted_iou)
        and metrics["motion_consistency"] >= float(gate.min_motion_consistency)
        and metrics["shape_consistency"] >= float(gate.min_shape_consistency)
    )


def _apply_suffix_paths(predicted: np.ndarray, selected: Sequence[_PathCandidate]) -> np.ndarray:
    output = np.asarray(predicted, dtype=int).copy()
    for path in selected:
        component_id = int(path.component_id)
        if component_id < 0 or component_id >= output.shape[0]:
            continue
        for edge in path.edges:
            _session_a, session_b, _roi_a, roi_b = edge.edge
            if session_b < output.shape[1] and output[component_id, session_b] < 0:
                output[component_id, session_b] = int(roi_b)
    return output


def _candidate_row(
    subject: str,
    path: _PathCandidate,
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    baseline_scores: Mapping[str, float | int],
    selected: bool,
    gate: CoherenceSuffixStitchGate,
) -> dict[str, float | int | str]:
    candidate_scores = dict(score_track_matrices(_apply_suffix_paths(predicted, (path,)), reference))
    delta = _score_delta(baseline_scores, candidate_scores)
    base_row = _path_row(subject, path, predicted, reference)
    metrics = _path_metrics(path)
    base_row.update(
        {
            "selected_by_gate": int(selected),
            "gate_pass": int(_passes_coherence_gate(path, predicted, reference, gate=gate)),
            "path_rank_under_existing_score": int(path.path_rank),
            "pairwise_tp_delta": int(delta["pairwise_true_positives"]),
            "pairwise_fp_delta": int(delta["pairwise_false_positives"]),
            "pairwise_fn_delta": int(delta["pairwise_false_negatives"]),
            "complete_tp_delta": int(delta["complete_track_true_positives"]),
            "complete_fp_delta": int(delta["complete_track_false_positives"]),
            "complete_fn_delta": int(delta["complete_track_false_negatives"]),
            "min_cell_probability": float(metrics["min_cell_probability"]),
            "min_area_ratio": float(metrics["min_area_ratio"]),
            "max_centroid_distance": float(metrics["max_centroid_distance"]),
            "min_shifted_iou": float(metrics["min_shifted_iou"]),
            "motion_consistency": float(metrics["motion_consistency"]),
            "shape_consistency": float(metrics["shape_consistency"]),
        }
    )
    return base_row


def _result_row(
    subject: str,
    baseline: Mapping[str, float | int],
    stitched: Mapping[str, float | int],
    selected: Sequence[_PathCandidate],
    candidate_rows: Sequence[Mapping[str, float | int | str]],
    *,
    gate: CoherenceSuffixStitchGate,
) -> dict[str, float | int | str]:
    del gate
    delta = _score_delta(baseline, stitched)
    selected_rows = [row for row in candidate_rows if int(row.get("selected_by_gate", 0)) > 0]
    pairwise_f1 = _f1_from_counts(
        stitched["pairwise_true_positives"],
        stitched["pairwise_false_positives"],
        stitched["pairwise_false_negatives"],
    )
    complete_track_f1 = _f1_from_counts(
        stitched["complete_track_true_positives"],
        stitched["complete_track_false_positives"],
        stitched["complete_track_false_negatives"],
    )
    return {
        "subject": subject,
        "selected_paths": int(len(selected)),
        "selected_gt_suffix_paths": int(sum(int(row.get("is_gt_suffix_path", 0)) for row in selected_rows)),
        "selected_non_gt_suffix_paths": int(sum(1 - int(row.get("is_gt_suffix_path", 0)) for row in selected_rows)),
        "candidate_paths": int(len(candidate_rows)),
        "pairwise_true_positives": int(stitched["pairwise_true_positives"]),
        "pairwise_false_positives": int(stitched["pairwise_false_positives"]),
        "pairwise_false_negatives": int(stitched["pairwise_false_negatives"]),
        "pairwise_f1": pairwise_f1,
        "pairwise_f1_micro": pairwise_f1,
        "complete_track_true_positives": int(stitched["complete_track_true_positives"]),
        "complete_track_false_positives": int(stitched["complete_track_false_positives"]),
        "complete_track_false_negatives": int(stitched["complete_track_false_negatives"]),
        "complete_track_f1": complete_track_f1,
        "complete_track_f1_micro": complete_track_f1,
        "pairwise_tp_delta": int(delta["pairwise_true_positives"]),
        "pairwise_fp_delta": int(delta["pairwise_false_positives"]),
        "pairwise_fn_delta": int(delta["pairwise_false_negatives"]),
        "complete_tp_delta": int(delta["complete_track_true_positives"]),
        "complete_fp_delta": int(delta["complete_track_false_positives"]),
        "complete_fn_delta": int(delta["complete_track_false_negatives"]),
        "selected_candidate_paths": ";".join(str(row.get("candidate_path", "")) for row in selected_rows),
    }


def _aggregate_result_row(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    gate: CoherenceSuffixStitchGate,
) -> dict[str, float | int | str]:
    del gate
    keys = (
        "pairwise_true_positives",
        "pairwise_false_positives",
        "pairwise_false_negatives",
        "complete_track_true_positives",
        "complete_track_false_positives",
        "complete_track_false_negatives",
        "pairwise_tp_delta",
        "pairwise_fp_delta",
        "pairwise_fn_delta",
        "complete_tp_delta",
        "complete_fp_delta",
        "complete_fn_delta",
        "selected_paths",
        "selected_gt_suffix_paths",
        "selected_non_gt_suffix_paths",
        "candidate_paths",
    )
    output = {"subject": "ALL"}
    for key in keys:
        output[key] = int(sum(int(row.get(key, 0)) for row in rows))
    output["pairwise_f1_micro"] = _f1_from_counts(
        output["pairwise_true_positives"],
        output["pairwise_false_positives"],
        output["pairwise_false_negatives"],
    )
    output["pairwise_f1"] = output["pairwise_f1_micro"]
    output["complete_track_f1_micro"] = _f1_from_counts(
        output["complete_track_true_positives"],
        output["complete_track_false_positives"],
        output["complete_track_false_negatives"],
    )
    output["complete_track_f1"] = output["complete_track_f1_micro"]
    output["selected_candidate_paths"] = ";".join(str(row.get("selected_candidate_paths", "")) for row in rows if str(row.get("selected_candidate_paths", "")))
    return output


def _path_metrics(path: _PathCandidate) -> dict[str, float]:
    edges = path.edges
    min_cell_probability = min(_nanmin_default((edge.cell_probability_a, edge.cell_probability_b), float("nan")) for edge in edges)
    return {
        "min_cell_probability": float(min_cell_probability),
        "min_area_ratio": float(_min_attr(edges, "area_ratio")),
        "max_centroid_distance": float(_max_attr(edges, "centroid_distance")),
        "min_shifted_iou": float(_min_attr(edges, "shifted_iou")),
        "motion_consistency": float(_motion_consistency(edges)),
        "shape_consistency": float(_mean_attr(edges, "area_ratio")),
    }


def _coherence_sort_key(path: _PathCandidate) -> tuple[float, ...]:
    metrics = _path_metrics(path)
    return (
        -metrics["motion_consistency"],
        -metrics["shape_consistency"],
        -metrics["min_cell_probability"],
        -metrics["min_area_ratio"],
        metrics["max_centroid_distance"],
        int(path.path_rank),
    )


def _would_reach_final_session(path: _PathCandidate, predicted: np.ndarray) -> bool:
    return bool(path.edges and path.edges[-1].edge[1] >= predicted.shape[1] - 1)


def _target_slot_occupied(path: _PathCandidate, predicted: np.ndarray) -> tuple[bool, ...]:
    output: list[bool] = []
    component_id = int(path.component_id)
    if component_id < 0 or component_id >= predicted.shape[0]:
        return tuple(True for _edge in path.edges)
    for edge in path.edges:
        _session_a, session_b, _roi_a, roi_b = edge.edge
        output.append(session_b >= predicted.shape[1] or (predicted[component_id, session_b] >= 0 and int(predicted[component_id, session_b]) != int(roi_b)))
    return tuple(output)


def _motion_consistency(edges: Sequence[Any]) -> float:
    distances = [float(edge.centroid_distance) for edge in edges if np.isfinite(float(edge.centroid_distance))]
    if len(distances) <= 1:
        return 1.0
    return float(1.0 / (1.0 + np.std(np.asarray(distances, dtype=float))))


def _score_delta(
    baseline: Mapping[str, float | int],
    candidate: Mapping[str, float | int],
) -> dict[str, int]:
    return {
        key: int(candidate[key]) - int(baseline[key])
        for key in (
            "pairwise_true_positives",
            "pairwise_false_positives",
            "pairwise_false_negatives",
            "complete_track_true_positives",
            "complete_track_false_positives",
            "complete_track_false_negatives",
        )
    }


def _f1_from_counts(tp: Any, fp: Any, fn: Any) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)
    if denominator == 0:
        return 1.0
    return float(2 * int(tp) / denominator)


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write rows as CSV or JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for coherence suffix-stitch."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-coherence-suffix-stitch",
        description="Run a coherence-gated suffix stitch after ComponentCleanup.",
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
    parser.add_argument("--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument("--suffix-path-length", type=int, default=2)
    parser.add_argument("--min-cell-probability", type=float, default=0.80)
    parser.add_argument("--min-area-ratio", type=float, default=0.80)
    parser.add_argument("--max-centroid-distance", type=float, default=6.0)
    parser.add_argument("--min-shifted-iou", type=float, default=0.30)
    parser.add_argument("--min-motion-consistency", type=float, default=0.50)
    parser.add_argument("--min-shape-consistency", type=float, default=0.82)
    parser.add_argument("--max-stitches-per-subject", type=int, default=1)
    parser.add_argument("--edge-top-k", type=int, default=25)
    parser.add_argument("--path-beam-width", type=int, default=100)
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
    parser.add_argument("--allow-track2p-as-reference-for-smoke-test", action="store_true")
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, default=None)
    parser.add_argument(
        "--aggregate-row",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include an ALL aggregate row in the score output.",
    )
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    """Run the coherence suffix-stitch what-if CLI."""

    args = (parser or build_arg_parser()).parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    gate = CoherenceSuffixStitchGate(
        suffix_path_length=int(args.suffix_path_length),
        min_cell_probability=float(args.min_cell_probability),
        min_area_ratio=float(args.min_area_ratio),
        max_centroid_distance=float(args.max_centroid_distance),
        min_shifted_iou=float(args.min_shifted_iou),
        min_motion_consistency=float(args.min_motion_consistency),
        min_shape_consistency=float(args.min_shape_consistency),
        max_stitches_per_subject=int(args.max_stitches_per_subject),
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
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    result = run_track2p_policy_coherence_suffix_stitch_whatif(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
        gate=gate,
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
    )
    result_rows = (
        result.result_rows
        if bool(args.aggregate_row)
        else tuple(row for row in result.result_rows if str(row.get("subject")) != "ALL")
    )
    write_rows(result_rows, args.output, output_format=args.format)
    if args.candidate_output is not None:
        write_rows(result.candidate_rows, args.candidate_output, output_format=args.format)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
