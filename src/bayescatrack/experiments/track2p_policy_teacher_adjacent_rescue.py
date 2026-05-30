"""Track2p-teacher adjacent rescue after component cleanup.

This is a deliberately narrow Track2p-teacher hybrid ablation.  It starts from
the frozen Track2pPolicy component-cleanup prediction and admits only adjacent
Track2p teacher edges that can extend an already existing seed-anchored
component without creating duplicate source/target observations or merging
components.  It does not use manual GT labels to choose edges.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
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
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_METHOD = (
    "track2p-policy-teacher-adjacent-rescue"
)


@dataclass(frozen=True)
class TeacherAdjacentRescueReport:
    """Prediction plus teacher-rescue diagnostic rows."""

    tracks: np.ndarray
    rows: tuple[dict[str, int | str], ...]


def run_track2p_policy_teacher_adjacent_rescue(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    allow_completing_rescue: bool = False,
) -> ComponentAuditOutput:
    """Run component cleanup followed by strict adjacent Track2p teacher rescue."""

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
    rescue_rows: list[dict[str, int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy teacher adjacent rescue requires independent "
                "manual GT references"
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
        teacher_full, _variant = _predict_subject_tracks(
            subject_dir, replace(policy_config, method="track2p-baseline")
        )
        rescue = apply_teacher_adjacent_rescue_edges(
            base_full,
            teacher_full,
            seed_session=policy_config.seed_session,
            allow_completing_rescue=allow_completing_rescue,
        )
        scores = _score_prediction_against_reference(
            rescue.tracks, reference, config=policy_config
        )
        applied = int(sum(int(row["applied"]) for row in rescue.rows))
        candidates = int(len(rescue.rows))
        scores = {
            **scores,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_teacher_adjacent_candidates": candidates,
            "track2p_teacher_adjacent_applied": applied,
            "track2p_teacher_adjacent_allow_completing_rescue": int(
                allow_completing_rescue
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant="Track2p-policy component cleanup + teacher adjacent rescue",
                method=cast(Any, TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        rescue_rows.extend(
            {
                **row,
                "subject": subject_dir.name,
                "threshold_method": str(threshold_method),
                "iou_distance_threshold": f"{float(iou_distance_threshold):g}",
                "cell_probability_threshold": f"{float(policy_config.cell_probability_threshold):g}",
                "transform_type": str(policy_config.transform_type),
            }
            for row in rescue.rows
        )
    return ComponentAuditOutput(tuple(results), tuple(rescue_rows))


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
        subject="",
        config=cleanup_config,
        track_ids=evaluated_track_ids,
        seed_session=config.seed_session,
    )
    return apply_weakest_bridge_splits(
        policy_full, _mark_applied_splits(audit_rows, apply_splits=True)
    )


def apply_teacher_adjacent_rescue_edges(
    predicted_track_matrix: Any,
    teacher_track_matrix: Any,
    *,
    seed_session: int = 0,
    allow_completing_rescue: bool = False,
) -> TeacherAdjacentRescueReport:
    """Extend seed-anchored components with conflict-free adjacent teacher edges."""

    output = _normalize_int_track_matrix(predicted_track_matrix)
    teacher = _normalize_int_track_matrix(teacher_track_matrix)
    rows: list[dict[str, int | str]] = []
    for edge, count in sorted(track_edge_counter(teacher).items()):
        for occurrence_index in range(int(count)):
            if track_edge_counter(output).get(edge, 0) > occurrence_index:
                continue
            output, row = _try_apply_teacher_edge(
                output,
                edge,
                seed_session=seed_session,
                allow_completing_rescue=allow_completing_rescue,
            )
            rows.append(
                {
                    **row,
                    "occurrence_index": int(occurrence_index),
                }
            )
    return TeacherAdjacentRescueReport(output, tuple(rows))


def _try_apply_teacher_edge(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    seed_session: int,
    allow_completing_rescue: bool = False,
) -> tuple[np.ndarray, dict[str, int | str]]:
    output = np.asarray(predicted, dtype=int).copy()
    session_a, session_b, roi_a, roi_b = edge
    row = {
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "applied": 0,
        "reason": "not_evaluated",
        "source_row": -1,
        "target_row": -1,
    }
    if session_b != session_a + 1:
        row["reason"] = "not_adjacent"
        return output, row
    source_rows = tuple(np.flatnonzero(output[:, session_a] == roi_a))
    target_rows = tuple(np.flatnonzero(output[:, session_b] == roi_b))
    if len(source_rows) != 1:
        row["reason"] = "missing_or_ambiguous_source"
        return output, row
    source_row = int(source_rows[0])
    row["source_row"] = source_row
    if seed_session < 0 or seed_session >= output.shape[1] or output[
        source_row, seed_session
    ] < 0:
        row["reason"] = "source_not_seed_anchored"
        return output, row
    if output[source_row, session_b] >= 0:
        row["reason"] = "source_has_target_conflict"
        return output, row
    if len(target_rows) > 0:
        row["target_row"] = int(target_rows[0])
        row["reason"] = "target_already_claimed"
        return output, row
    candidate_row = output[source_row].copy()
    candidate_row[session_b] = roi_b
    if not allow_completing_rescue and np.all(candidate_row >= 0):
        row["reason"] = "would_complete_track"
        return output, row
    output[source_row, session_b] = roi_b
    row["applied"] = 1
    row["reason"] = "accepted_insert_target"
    return output, row


def write_rescue_rows(
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
        prog="bayescatrack benchmark track2p-policy-teacher-adjacent-rescue",
        description=(
            "Run Track2pPolicy component cleanup and then extend seed-anchored "
            "components with conflict-free adjacent Track2p teacher edges."
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
        "--allow-completing-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow teacher rescue edges that would turn an incomplete "
            "seed-anchored component into a complete row."
        ),
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
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    output = run_track2p_policy_teacher_adjacent_rescue(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
        cleanup_config=cleanup_config,
        allow_completing_rescue=args.allow_completing_rescue,
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.diagnostics_output is not None:
        write_rescue_rows(
            output.component_rows,
            args.diagnostics_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
