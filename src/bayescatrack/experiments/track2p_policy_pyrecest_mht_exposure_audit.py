"""Audit PyRecEst residual-MHT candidate exposure without manual ground truth.

This diagnostic runs the non-teacher CoherenceSuffixStitch starting point over
every Track2p-style subject under the data root, enumerates the same label-free
residual-MHT candidate pockets as the cleanup runner, and reports how often the
selector would fire.  It intentionally does not load manual GT, score against
manual references, or emit GT/status/delta columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from bayescatrack.evaluation.complete_track_scores import score_track_matrices
from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_stitch_whatif as suffix,
)
from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup
from bayescatrack.experiments import track2p_policy_growth_veto_whatif as veto
from bayescatrack.experiments import (
    track2p_policy_pyrecest_residual_mht_cleanup as residual_mht,
)
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _load_subject_sessions,
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
from bayescatrack.experiments.track2p_policy_suffix_stitch_ranking_audit import (
    _FeatureCache,
    _ranked_suffix_paths,
)

METHOD = "track2p-policy-pyrecest-mht-exposure-audit"


@dataclass(frozen=True)
class PyRecEstMHTExposureAuditResult:
    """Per-subject exposure rows plus optional candidate details."""

    summary_rows: tuple[dict[str, Any], ...]
    detail_rows: tuple[dict[str, Any], ...]


def run_track2p_policy_pyrecest_mht_exposure_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    suffix_gate: suffix.CoherenceSuffixStitchGate | None = None,
    edge_top_k: int = 25,
    path_beam_width: int = 100,
    anchor_min_registered_iou: float = 0.50,
    anchor_min_shifted_iou: float = 0.30,
    anchor_min_cell_probability: float = 0.80,
    growth_veto_gate: cleanup.GrowthVetoGate | None = None,
    mht_options: residual_mht.PyRecEstResidualMHTOptions | None = None,
    progress: bool = False,
) -> PyRecEstMHTExposureAuditResult:
    """Return label-free PyRecEst MHT exposure over all discovered subjects."""

    edge_top_k = suffix._positive_int_value(edge_top_k, name="edge_top_k")
    path_beam_width = suffix._positive_int_value(path_beam_width, name="path_beam_width")
    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    suffix_gate = suffix_gate or suffix.CoherenceSuffixStitchGate()
    growth_veto_gate = growth_veto_gate or cleanup.GrowthVetoGate()
    mht_options = mht_options or residual_mht.PyRecEstResidualMHTOptions()

    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(f"No Track2p-style subject directories found under {policy_config.data}")

    states = [
        _subject_state_no_gt(
            subject_dir,
            config=policy_config,
            cleanup_config=cleanup_config,
            suffix_gate=suffix_gate,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            edge_top_k=int(edge_top_k),
            path_beam_width=int(path_beam_width),
            anchor_min_registered_iou=float(anchor_min_registered_iou),
            anchor_min_shifted_iou=float(anchor_min_shifted_iou),
            anchor_min_cell_probability=float(anchor_min_cell_probability),
            progress=progress,
        )
        for subject_dir in subject_dirs
    ]
    global_baseline_scores = veto._global_scores(state.baseline_scores for state in states)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for state in states:
        _log_progress(progress, f"{METHOD}: {state.subject}: enumerate candidates")
        edge_rows = veto._accepted_edge_rows(
            state,
            global_baseline_scores=global_baseline_scores,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            cell_probability_threshold=float(policy_config.cell_probability_threshold),
            transform_type=str(policy_config.transform_type),
        )
        edge_rows = cleanup._augment_growth_veto_candidate_shifted_iou(
            edge_rows,
            state.sessions,
            gate=growth_veto_gate,
            n_sessions=int(state.combined.shape[1]),
        )
        candidate_rows = residual_mht._candidate_rows(
            edge_rows,
            gate=growth_veto_gate,
            options=mht_options,
            n_sessions=int(state.combined.shape[1]),
        )
        selected_rows, hypothesis_count = _select_candidate_rows(
            candidate_rows,
            base_tracks=state.combined,
            growth_veto_gate=growth_veto_gate,
            options=mht_options,
        )
        _edited_tracks, applied_keys = residual_mht._apply_selected_growth_veto_rows(
            state.combined,
            selected_rows,
            gate=growth_veto_gate,
        )
        summary_rows.append(
            _exposure_row(
                state.subject,
                candidate_rows,
                selected_rows,
                applied_keys,
                accepted_edges=len(edge_rows),
                hypothesis_count=hypothesis_count,
            )
        )
        detail_rows.extend(
            _detail_rows(
                state.subject,
                candidate_rows,
                selected_rows,
                applied_keys,
                options=mht_options,
            )
        )
    summary_rows.append(_aggregate_row(summary_rows))
    return PyRecEstMHTExposureAuditResult(tuple(summary_rows), tuple(detail_rows))


def _subject_state_no_gt(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    suffix_gate: suffix.CoherenceSuffixStitchGate,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    edge_top_k: int,
    path_beam_width: int,
    anchor_min_registered_iou: float,
    anchor_min_shifted_iou: float,
    anchor_min_cell_probability: float,
    progress: bool,
) -> veto._SubjectState:
    sessions = _load_subject_sessions(subject_dir, config)
    n_sessions = len(sessions)
    empty_reference = np.empty((0, n_sessions), dtype=int)
    _log_progress(progress, f"{METHOD}: {subject_dir.name}: sessions loaded")

    policy_prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    policy_full = _normalize_int_track_matrix(policy_prediction.tracks)
    audit_rows = component_audit_rows(
        policy_full,
        empty_reference,
        sessions=sessions,
        diagnostics=policy_prediction.diagnostics,
        subject=subject_dir.name,
        config=cleanup_config,
        track_ids=tuple(range(policy_full.shape[0])),
        seed_session=config.seed_session,
    )
    cleaned = apply_weakest_bridge_splits(policy_full, _mark_applied_splits(audit_rows, apply_splits=True))
    cleaned = veto._pad_track_matrix(veto._as_track_matrix(cleaned), width=n_sessions)
    _log_progress(progress, f"{METHOD}: {subject_dir.name}: cleanup ready")

    feature_cache = _FeatureCache(
        sessions=sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(config.cell_probability_threshold),
        matrices={},
    )
    paths = _ranked_suffix_paths(
        cleaned,
        empty_reference,
        subject=subject_dir.name,
        feature_cache=feature_cache,
        max_suffix_length=int(suffix_gate.suffix_path_length),
        edge_top_k=int(edge_top_k),
        path_beam_width=int(path_beam_width),
    )
    selected = suffix._select_paths(paths, cleaned, gate=suffix_gate)
    edge_features = veto._policy_feature_index_from_diagnostics(sessions, policy_prediction.diagnostics)
    edge_features.update(veto._suffix_feature_index(selected))
    stitched = veto._pad_track_matrix(
        veto._as_track_matrix(suffix._apply_suffix_paths(cleaned, selected)),
        width=n_sessions,
    )
    _log_progress(
        progress,
        f"{METHOD}: {subject_dir.name}: suffix candidates={len(paths)} selected={len(selected)}",
    )

    anchor_edges = veto._anchor_edges_from_policy_diagnostics(
        sessions,
        feature_cache=feature_cache,
        diagnostics=policy_prediction.diagnostics,
        track2p=stitched,
        component_cleanup=cleaned,
        combined=stitched,
        min_registered_iou=float(anchor_min_registered_iou),
        min_shifted_iou=float(anchor_min_shifted_iou),
        min_cell_probability=float(anchor_min_cell_probability),
    )
    growth_models = veto._growth_models_by_pair(sessions, anchor_edges)
    growth_context = veto._growth_feature_context(sessions, stitched, anchor_edges)
    return veto._SubjectState(
        subject=subject_dir.name,
        sessions=sessions,
        policy=veto._pad_track_matrix(veto._as_track_matrix(policy_full), width=n_sessions),
        component_cleanup=cleaned,
        coherence_suffix=stitched,
        teacher=stitched,
        combined=stitched,
        reference=empty_reference,
        edge_features=edge_features,
        anchor_edges=anchor_edges,
        growth_models=growth_models,
        growth_context=growth_context,
        baseline_scores=dict(score_track_matrices(stitched, empty_reference)),
    )


def _select_candidate_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    base_tracks: np.ndarray,
    growth_veto_gate: cleanup.GrowthVetoGate,
    options: residual_mht.PyRecEstResidualMHTOptions,
) -> tuple[list[Mapping[str, Any]], int]:
    if options.selection_mode == "deterministic-gating":
        return (
            residual_mht._select_deterministic_gating_rows(
                candidate_rows,
                options=options,
            ),
            0,
        )

    pyrecest_candidates = [residual_mht._to_pyrecest_candidate(row, options=options) for row in candidate_rows]
    pyrecest_config = residual_mht.ResidualMHTConfig(
        max_edits=int(options.max_edits_per_subject),
        max_hypotheses=int(options.max_hypotheses),
        edit_penalty=float(options.edit_penalty),
        score_threshold=float(options.score_threshold),
        include_empty=True,
    )
    hypotheses = residual_mht.enumerate_residual_hypotheses(
        pyrecest_candidates,
        config=pyrecest_config,
    )
    if options.selection_mode == "global-rescore":
        selected_hypothesis, _selected_objective = residual_mht._select_residual_hypothesis_global_rescore(
            hypotheses,
            candidate_rows,
            base_tracks=base_tracks,
            gate=growth_veto_gate,
            options=options,
        )
    else:
        selected_hypothesis = residual_mht.select_residual_hypothesis(
            pyrecest_candidates,
            config=pyrecest_config,
        )
    selected_ids = set(selected_hypothesis.candidate_ids)
    return (
        [row for row in candidate_rows if str(row["pyrecest_candidate_id"]) in selected_ids],
        int(len(hypotheses)),
    )


def _exposure_row(
    subject: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    applied_keys: set[tuple[int, int, int, int, int]],
    *,
    accepted_edges: int,
    hypothesis_count: int,
) -> dict[str, Any]:
    candidates_by_family = Counter(str(row.get("pyrecest_candidate_family", "")) for row in candidate_rows)
    selected_by_family = Counter(str(row.get("pyrecest_candidate_family", "")) for row in selected_rows)
    applied_rows = [row for row in selected_rows if cleanup._edge_row_key(row) in set(applied_keys)]
    applied_by_family = Counter(str(row.get("pyrecest_candidate_family", "")) for row in applied_rows)
    return {
        "subject": subject,
        "accepted_edges": int(accepted_edges),
        "growth_pocket_candidates": int(candidates_by_family["growth_veto"]),
        "high_overlap_low_motion_pocket_candidates": int(candidates_by_family["high_overlap_low_motion"]),
        "compact_low_overlap_pocket_candidates": int(candidates_by_family["compact_low_overlap"]),
        "total_mht_candidates": int(len(candidate_rows)),
        "selected_mht_edits": int(len(selected_rows)),
        "applied_mht_edits": int(len(applied_rows)),
        "selected_edits_per_subject": int(len(selected_rows)),
        "growth_pocket_selected": int(selected_by_family["growth_veto"]),
        "high_overlap_low_motion_pocket_selected": int(selected_by_family["high_overlap_low_motion"]),
        "compact_low_overlap_pocket_selected": int(selected_by_family["compact_low_overlap"]),
        "growth_pocket_applied": int(applied_by_family["growth_veto"]),
        "high_overlap_low_motion_pocket_applied": int(applied_by_family["high_overlap_low_motion"]),
        "compact_low_overlap_pocket_applied": int(applied_by_family["compact_low_overlap"]),
        "mht_hypotheses": int(hypothesis_count),
        "selected_candidate_ids": ";".join(str(row.get("pyrecest_candidate_id", "")) for row in selected_rows),
    }


def _aggregate_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_subject = tuple(row for row in rows if str(row.get("subject", "")) != "ALL")
    sum_keys = (
        "accepted_edges",
        "growth_pocket_candidates",
        "high_overlap_low_motion_pocket_candidates",
        "compact_low_overlap_pocket_candidates",
        "total_mht_candidates",
        "selected_mht_edits",
        "applied_mht_edits",
        "growth_pocket_selected",
        "high_overlap_low_motion_pocket_selected",
        "compact_low_overlap_pocket_selected",
        "growth_pocket_applied",
        "high_overlap_low_motion_pocket_applied",
        "compact_low_overlap_pocket_applied",
        "mht_hypotheses",
    )
    output: dict[str, Any] = {"subject": "ALL"}
    for key in sum_keys:
        output[key] = int(sum(int(row.get(key, 0)) for row in per_subject))
    output["selected_edits_per_subject"] = int(max((int(row.get("selected_mht_edits", 0)) for row in per_subject), default=0))
    output["selected_candidate_ids"] = ";".join(str(row.get("selected_candidate_ids", "")) for row in per_subject if str(row.get("selected_candidate_ids", "")))
    return output


def _detail_rows(
    subject: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    applied_keys: set[tuple[int, int, int, int, int]],
    *,
    options: residual_mht.PyRecEstResidualMHTOptions,
) -> list[dict[str, Any]]:
    selected_ids = {str(row.get("pyrecest_candidate_id", "")) for row in selected_rows}
    applied_set = set(applied_keys)
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        min_cell_probability = min(
            residual_mht._finite_float(row.get("cell_probability_a"), float("nan")),
            residual_mht._finite_float(row.get("cell_probability_b"), float("nan")),
        )
        rows.append(
            {
                "subject": subject,
                "pyrecest_candidate_id": str(row.get("pyrecest_candidate_id", "")),
                "pyrecest_candidate_family": str(row.get("pyrecest_candidate_family", "")),
                "selected_by_pyrecest_mht": int(str(row.get("pyrecest_candidate_id", "")) in selected_ids),
                "applied_by_pyrecest_mht": int(cleanup._edge_row_key(row) in applied_set),
                "session_a": int(row.get("session_a", -1)),
                "roi_a": int(row.get("roi_a", -1)),
                "session_b": int(row.get("session_b", -1)),
                "roi_b": int(row.get("roi_b", -1)),
                "pyrecest_candidate_score": residual_mht._candidate_score(
                    row,
                    options=options,
                ),
                "growth_residual_mahalanobis": residual_mht._finite_float(row.get("growth_residual_mahalanobis"), float("nan")),
                "growth_residual": residual_mht._finite_float(row.get("growth_residual"), float("nan")),
                "registered_iou": residual_mht._finite_float(row.get("registered_iou"), float("nan")),
                "shifted_iou": residual_mht._finite_float(row.get("shifted_iou"), float("nan")),
                "min_cell_probability": float(min_cell_probability),
                "row_rank": int(row.get("row_rank", -1)),
                "column_rank": int(row.get("column_rank", -1)),
            }
        )
    return rows


def _optional_float_arg(value: str) -> float | None:
    if str(value).lower() in {"none", "null", ""}:
        return None
    return float(value)


def _optional_int_arg(value: str) -> int | None:
    if str(value).lower() in {"none", "null", ""}:
        return None
    return int(value)


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(list(rows), indent=2) + "\n")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        "subject",
        "growth_pocket_candidates",
        "high_overlap_low_motion_pocket_candidates",
        "selected_mht_edits",
        "applied_mht_edits",
        "selected_edits_per_subject",
    )
    lines = [
        "| subject | growth pocket candidates | high-overlap/low-motion candidates | selected MHT edits | applied MHT edits | selected edits per subject |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"bayescatrack benchmark {METHOD}",
        description=("Run a no-GT exposure audit for PyRecEst residual-MHT candidate pockets over all Track2p-style subjects under --data."),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument("--input-format", choices=("auto", "suite2p", "npy"), default="suite2p")
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    parser.add_argument("--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE)
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
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--suffix-path-length", type=residual_mht._positive_int_arg, default=2)
    parser.add_argument("--min-cell-probability", type=float, default=0.80)
    parser.add_argument("--min-area-ratio", type=float, default=0.80)
    parser.add_argument("--max-centroid-distance", type=float, default=6.0)
    parser.add_argument("--min-shifted-iou", type=float, default=0.30)
    parser.add_argument("--min-motion-consistency", type=float, default=0.50)
    parser.add_argument("--min-shape-consistency", type=float, default=0.82)
    parser.add_argument("--max-stitches-per-subject", type=residual_mht._positive_int_arg, default=1)
    parser.add_argument("--edge-top-k", type=residual_mht._positive_int_arg, default=25)
    parser.add_argument("--path-beam-width", type=residual_mht._positive_int_arg, default=100)
    parser.add_argument("--anchor-min-registered-iou", type=float, default=0.50)
    parser.add_argument("--anchor-min-shifted-iou", type=float, default=0.30)
    parser.add_argument("--anchor-min-cell-probability", type=float, default=0.80)
    parser.add_argument("--min-growth-residual-mahalanobis", type=float, default=20.0)
    parser.add_argument("--min-growth-residual", type=float, default=2.50)
    parser.add_argument("--min-veto-registered-iou", type=float, default=0.45)
    parser.add_argument("--max-veto-registered-iou", type=_optional_float_arg, default=0.60)
    parser.add_argument("--min-veto-shifted-iou", type=float, default=0.60)
    parser.add_argument("--max-veto-shifted-iou", type=_optional_float_arg, default=0.80)
    parser.add_argument("--min-veto-cell-probability", type=float, default=0.50)
    parser.add_argument("--max-veto-min-cell-probability", type=_optional_float_arg, default=0.65)
    parser.add_argument("--max-veto-local-neighbor-distortion", type=_optional_float_arg, default=None)
    parser.add_argument("--min-veto-anchor-count", type=int, default=2)
    parser.add_argument("--min-veto-complete-component-size", type=_optional_int_arg, default=None)
    parser.add_argument("--max-veto-row-rank", type=int, default=1)
    parser.add_argument("--max-veto-column-rank", type=int, default=1)
    parser.add_argument("--require-veto-not-suffix-edge", action="store_true")
    parser.add_argument("--require-veto-terminal-edge", action="store_true")
    parser.add_argument("--require-veto-last-session-edge", action="store_true")
    parser.add_argument("--require-veto-complete-component", action="store_true")
    parser.add_argument("--mht-candidate-top-k", type=residual_mht._positive_int_arg, default=8)
    parser.add_argument("--mht-max-edits-per-subject", type=residual_mht._positive_int_arg, default=2)
    parser.add_argument("--mht-max-hypotheses", type=residual_mht._positive_int_arg, default=32)
    parser.add_argument("--mht-edit-penalty", type=float, default=0.25)
    parser.add_argument("--mht-score-threshold", type=float, default=1.0)
    parser.add_argument(
        "--mht-selection-mode",
        choices=("additive", "global-rescore", "deterministic-gating"),
        default="additive",
    )
    parser.add_argument("--mht-fragmentation-penalty", type=float, default=0.5)
    parser.add_argument(
        "--mht-min-meaningful-track-length",
        type=residual_mht._positive_int_arg,
        default=2,
    )
    parser.add_argument(
        "--mht-include-high-overlap-low-motion-candidates",
        action="store_true",
    )
    parser.add_argument("--mht-high-overlap-min-registered-iou", type=float, default=0.85)
    parser.add_argument("--mht-high-overlap-max-growth-residual", type=float, default=0.50)
    parser.add_argument(
        "--mht-high-overlap-min-growth-residual-mahalanobis",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--mht-high-overlap-min-cell-probability",
        type=_optional_float_arg,
        default=None,
    )
    parser.add_argument("--mht-high-overlap-score-bonus", type=float, default=2.0)
    parser.add_argument("--mht-include-compact-low-overlap-candidates", action="store_true")
    parser.add_argument("--mht-compact-min-registered-iou", type=float, default=0.30)
    parser.add_argument("--mht-compact-max-registered-iou", type=float, default=0.55)
    parser.add_argument("--mht-compact-min-growth-residual", type=float, default=0.50)
    parser.add_argument("--mht-compact-max-growth-residual", type=float, default=2.50)
    parser.add_argument("--mht-compact-min-growth-residual-mahalanobis", type=float, default=2.0)
    parser.add_argument("--mht-compact-max-growth-residual-mahalanobis", type=float, default=6.0)
    parser.add_argument("--mht-compact-min-cell-probability", type=float, default=0.75)
    parser.add_argument("--mht-compact-component-size", type=_optional_int_arg, default=4)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def _log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        plane_name=args.plane_name,
        seed_session=0,
        transform_type=args.transform_type,
        include_behavior=False,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    result = run_track2p_policy_pyrecest_mht_exposure_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=ComponentCleanupConfig(
            split_risk_threshold=args.split_risk_threshold,
            split_penalty=args.split_penalty,
            min_side_observations=args.min_side_observations,
            require_complete_track=args.require_complete_track,
        ),
        suffix_gate=suffix.CoherenceSuffixStitchGate(
            suffix_path_length=int(args.suffix_path_length),
            min_cell_probability=float(args.min_cell_probability),
            min_area_ratio=float(args.min_area_ratio),
            max_centroid_distance=float(args.max_centroid_distance),
            min_shifted_iou=float(args.min_shifted_iou),
            min_motion_consistency=float(args.min_motion_consistency),
            min_shape_consistency=float(args.min_shape_consistency),
            max_stitches_per_subject=int(args.max_stitches_per_subject),
        ),
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
        anchor_min_registered_iou=float(args.anchor_min_registered_iou),
        anchor_min_shifted_iou=float(args.anchor_min_shifted_iou),
        anchor_min_cell_probability=float(args.anchor_min_cell_probability),
        growth_veto_gate=cleanup.GrowthVetoGate(
            min_growth_residual_mahalanobis=float(args.min_growth_residual_mahalanobis),
            min_growth_residual=float(args.min_growth_residual),
            min_registered_iou=float(args.min_veto_registered_iou),
            min_shifted_iou=float(args.min_veto_shifted_iou),
            max_registered_iou=args.max_veto_registered_iou,
            max_shifted_iou=args.max_veto_shifted_iou,
            min_cell_probability=float(args.min_veto_cell_probability),
            max_min_cell_probability=args.max_veto_min_cell_probability,
            max_local_neighbor_distortion=args.max_veto_local_neighbor_distortion,
            min_anchor_count=int(args.min_veto_anchor_count),
            min_complete_component_size=args.min_veto_complete_component_size,
            max_row_rank=int(args.max_veto_row_rank),
            max_column_rank=int(args.max_veto_column_rank),
            require_not_suffix_edge=bool(args.require_veto_not_suffix_edge),
            require_terminal_edge=bool(args.require_veto_terminal_edge),
            require_last_session_edge=bool(args.require_veto_last_session_edge),
            require_complete_component=bool(args.require_veto_complete_component),
            max_vetoes_per_subject=int(args.mht_max_edits_per_subject),
        ),
        mht_options=residual_mht.PyRecEstResidualMHTOptions(
            candidate_top_k=int(args.mht_candidate_top_k),
            max_edits_per_subject=int(args.mht_max_edits_per_subject),
            max_hypotheses=int(args.mht_max_hypotheses),
            edit_penalty=float(args.mht_edit_penalty),
            score_threshold=float(args.mht_score_threshold),
            selection_mode=cast(
                residual_mht.ResidualSelectionMode,
                args.mht_selection_mode,
            ),
            fragmentation_penalty=float(args.mht_fragmentation_penalty),
            min_meaningful_track_length=int(args.mht_min_meaningful_track_length),
            include_high_overlap_low_motion=bool(args.mht_include_high_overlap_low_motion_candidates),
            high_overlap_min_registered_iou=float(args.mht_high_overlap_min_registered_iou),
            high_overlap_max_growth_residual=float(args.mht_high_overlap_max_growth_residual),
            high_overlap_min_growth_residual_mahalanobis=float(args.mht_high_overlap_min_growth_residual_mahalanobis),
            high_overlap_min_cell_probability=args.mht_high_overlap_min_cell_probability,
            high_overlap_score_bonus=float(args.mht_high_overlap_score_bonus),
            include_compact_low_overlap=bool(args.mht_include_compact_low_overlap_candidates),
            compact_min_registered_iou=float(args.mht_compact_min_registered_iou),
            compact_max_registered_iou=float(args.mht_compact_max_registered_iou),
            compact_min_growth_residual=float(args.mht_compact_min_growth_residual),
            compact_max_growth_residual=float(args.mht_compact_max_growth_residual),
            compact_min_growth_residual_mahalanobis=float(args.mht_compact_min_growth_residual_mahalanobis),
            compact_max_growth_residual_mahalanobis=float(args.mht_compact_max_growth_residual_mahalanobis),
            compact_min_cell_probability=float(args.mht_compact_min_cell_probability),
            compact_component_size=args.mht_compact_component_size,
        ),
        progress=bool(args.progress),
    )
    write_rows(result.summary_rows, args.output, output_format=args.format)
    if args.details_output is not None:
        write_rows(result.detail_rows, args.details_output, output_format=args.format)
    if args.markdown_output is not None:
        write_markdown_summary(result.summary_rows, args.markdown_output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
