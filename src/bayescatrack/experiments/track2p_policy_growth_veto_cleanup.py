"""Apply a strict growth-field veto after CoherenceSuffixTeacherRescue.

The growth-veto what-if audit found one high-residual complete-FP candidate after
the current Track2p-teacher-assisted lead row.  That audit is label-aware in its
diagnostic columns, but the signal itself is label-free: a fitted per-session
growth field can mark an accepted adjacent edge as an extreme outlier.

This module turns that diagnostic into a deliberately narrow benchmark row.  It
starts from the same CoherenceSuffixTeacherRescue state as the what-if audit and
then splits at accepted adjacent edges that pass a hard, label-free gate:

* very high growth-field Mahalanobis residual;
* bounded local registered/shifted ROI evidence, so we do not split malformed
  or unmeasured edges, but can optionally avoid high-confidence local matches
  that are plausible true terminal continuations;
* sufficient endpoint cell probability, plus an optional upper bound on the
  weaker endpoint cell probability for weak-local-evidence veto candidates;
* optionally terminal/last-session and complete-component structure.

The defaults target the specific non-GT feature pocket identified by the audit:
one extreme terminal continuation in a complete component.  Manual-GT labels are
used only for final scoring and optional diagnostic columns inherited from the
what-if rows; they are not used to select vetoes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.shifted_overlap import _pairwise_shifted_iou_from_support
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_stitch_whatif as suffix,
)
from bayescatrack.experiments import track2p_policy_growth_veto_whatif as veto
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import _roi_indices
from bayescatrack.track2p_registration import register_plane_pair

METHOD = "track2p-policy-growth-veto-cleanup"
COHERENCE_SUFFIX_METHOD = "track2p-policy-coherence-suffix-growth-veto-cleanup"
GrowthVetoPredictionBase = Literal["teacher-rescue", "coherence-suffix"]


@dataclass(frozen=True)
class GrowthVetoGate:
    """Label-free gate for applying one-edge growth-veto splits."""

    min_growth_residual_mahalanobis: float = 20.0
    min_growth_residual: float = 2.5
    min_registered_iou: float = 0.45
    min_shifted_iou: float = 0.60
    max_registered_iou: float | None = 0.60
    max_shifted_iou: float | None = 0.80
    min_cell_probability: float = 0.50
    max_min_cell_probability: float | None = 0.65
    max_local_neighbor_distortion: float | None = 0.05
    min_anchor_count: int = 0
    min_complete_component_size: int | None = None
    max_row_rank: int = 1
    max_column_rank: int = 1
    require_not_suffix_edge: bool = True
    require_terminal_edge: bool = True
    require_last_session_edge: bool = True
    require_complete_component: bool = True
    max_vetoes_per_subject: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_anchor_count",
            suffix._nonnegative_int_value(
                self.min_anchor_count, name="min_anchor_count"
            ),
        )
        if self.min_complete_component_size is not None:
            object.__setattr__(
                self,
                "min_complete_component_size",
                suffix._nonnegative_int_value(
                    self.min_complete_component_size,
                    name="min_complete_component_size",
                ),
            )
        object.__setattr__(
            self,
            "max_row_rank",
            suffix._positive_int_value(self.max_row_rank, name="max_row_rank"),
        )
        object.__setattr__(
            self,
            "max_column_rank",
            suffix._positive_int_value(self.max_column_rank, name="max_column_rank"),
        )
        object.__setattr__(
            self,
            "max_vetoes_per_subject",
            suffix._positive_int_value(
                self.max_vetoes_per_subject, name="max_vetoes_per_subject"
            ),
        )


@dataclass(frozen=True)
class GrowthVetoCleanupResult:
    """Benchmark rows plus growth-veto diagnostic ledger."""

    results: tuple[SubjectBenchmarkResult, ...]
    edge_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


def run_track2p_policy_growth_veto_cleanup(
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
    growth_veto_gate: GrowthVetoGate | None = None,
    prediction_base: GrowthVetoPredictionBase = "teacher-rescue",
    progress: bool = False,
) -> GrowthVetoCleanupResult:
    """Run a coherence-suffix-family prediction followed by growth-veto splits."""

    edge_top_k = suffix._positive_int_value(edge_top_k, name="edge_top_k")
    path_beam_width = suffix._positive_int_value(
        path_beam_width, name="path_beam_width"
    )
    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    suffix_gate = suffix_gate or suffix.CoherenceSuffixStitchGate()
    growth_veto_gate = growth_veto_gate or GrowthVetoGate()
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    states = [
        veto._subject_state(
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
            prediction_base=prediction_base,
            progress=progress,
        )
        for subject_dir in subject_dirs
    ]
    global_baseline_scores = veto._global_scores(
        state.baseline_scores for state in states
    )

    results: list[SubjectBenchmarkResult] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for state in states:
        edge_rows = veto._accepted_edge_rows(
            state,
            global_baseline_scores=global_baseline_scores,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            cell_probability_threshold=float(policy_config.cell_probability_threshold),
            transform_type=policy_config.transform_type,
        )
        edge_rows = _augment_growth_veto_candidate_shifted_iou(
            edge_rows,
            state.sessions,
            gate=growth_veto_gate,
            n_sessions=int(state.reference.shape[1]),
        )
        selected = _selected_growth_veto_rows(
            edge_rows,
            gate=growth_veto_gate,
            n_sessions=int(state.reference.shape[1]),
        )
        vetoed_tracks, applied_keys = _apply_growth_veto_rows(
            state.combined,
            selected,
            gate=growth_veto_gate,
        )
        scores = dict(score_track_matrices(vetoed_tracks, state.reference))
        scores.update(
            {
                "track2p_growth_veto_candidates": int(len(edge_rows)),
                "track2p_growth_veto_selected": int(len(selected)),
                "track2p_growth_veto_applied": int(len(applied_keys)),
                "track2p_growth_veto_min_mahalanobis": float(
                    growth_veto_gate.min_growth_residual_mahalanobis
                ),
                "track2p_growth_veto_min_residual": float(
                    growth_veto_gate.min_growth_residual
                ),
                "track2p_growth_veto_min_registered_iou": float(
                    growth_veto_gate.min_registered_iou
                ),
                "track2p_growth_veto_min_shifted_iou": float(
                    growth_veto_gate.min_shifted_iou
                ),
                "track2p_growth_veto_max_registered_iou": (
                    float(growth_veto_gate.max_registered_iou)
                    if growth_veto_gate.max_registered_iou is not None
                    else float("nan")
                ),
                "track2p_growth_veto_max_shifted_iou": (
                    float(growth_veto_gate.max_shifted_iou)
                    if growth_veto_gate.max_shifted_iou is not None
                    else float("nan")
                ),
                "track2p_growth_veto_min_cell_probability": float(
                    growth_veto_gate.min_cell_probability
                ),
                "track2p_growth_veto_max_min_cell_probability": (
                    float(growth_veto_gate.max_min_cell_probability)
                    if growth_veto_gate.max_min_cell_probability is not None
                    else float("nan")
                ),
                "track2p_growth_veto_max_local_neighbor_distortion": (
                    float(growth_veto_gate.max_local_neighbor_distortion)
                    if growth_veto_gate.max_local_neighbor_distortion is not None
                    else float("nan")
                ),
                "track2p_growth_veto_min_anchor_count": int(
                    growth_veto_gate.min_anchor_count
                ),
                "track2p_growth_veto_min_complete_component_size": int(
                    growth_veto_gate.min_complete_component_size
                    if growth_veto_gate.min_complete_component_size is not None
                    else 0
                ),
                "track2p_growth_veto_max_row_rank": int(growth_veto_gate.max_row_rank),
                "track2p_growth_veto_max_column_rank": int(
                    growth_veto_gate.max_column_rank
                ),
                "track2p_growth_veto_require_not_suffix_edge": int(
                    growth_veto_gate.require_not_suffix_edge
                ),
                "track2p_growth_veto_require_terminal_edge": int(
                    growth_veto_gate.require_terminal_edge
                ),
                "track2p_growth_veto_require_last_session_edge": int(
                    growth_veto_gate.require_last_session_edge
                ),
                "track2p_growth_veto_require_complete_component": int(
                    growth_veto_gate.require_complete_component
                ),
                "track2p_growth_veto_max_vetoes_per_subject": int(
                    growth_veto_gate.max_vetoes_per_subject
                ),
            }
        )
        results.append(
            SubjectBenchmarkResult(
                subject=state.subject,
                variant=_variant_for_prediction_base(prediction_base),
                method=cast(Any, _method_for_prediction_base(prediction_base)),
                scores=scores,
                n_sessions=int(state.reference.shape[1]),
                reference_source=GROUND_TRUTH_REFERENCE_SOURCE,
            )
        )
        applied_set = set(applied_keys)
        selected_set = {_edge_row_key(row) for row in selected}
        for row in edge_rows:
            key = _edge_row_key(row)
            reason = growth_veto_gate_reason(
                row,
                growth_veto_gate,
                n_sessions=int(state.reference.shape[1]),
            )
            diagnostic_rows.append(
                {
                    **row,
                    "selected_by_growth_veto": int(key in selected_set),
                    "applied_by_growth_veto": int(key in applied_set),
                    "growth_veto_reason": reason,
                    "growth_veto_min_mahalanobis": float(
                        growth_veto_gate.min_growth_residual_mahalanobis
                    ),
                    "growth_veto_min_residual": float(
                        growth_veto_gate.min_growth_residual
                    ),
                    "growth_veto_min_registered_iou": float(
                        growth_veto_gate.min_registered_iou
                    ),
                    "growth_veto_min_shifted_iou": float(
                        growth_veto_gate.min_shifted_iou
                    ),
                    "growth_veto_max_registered_iou": (
                        float(growth_veto_gate.max_registered_iou)
                        if growth_veto_gate.max_registered_iou is not None
                        else float("nan")
                    ),
                    "growth_veto_max_shifted_iou": (
                        float(growth_veto_gate.max_shifted_iou)
                        if growth_veto_gate.max_shifted_iou is not None
                        else float("nan")
                    ),
                    "growth_veto_min_cell_probability": float(
                        growth_veto_gate.min_cell_probability
                    ),
                    "growth_veto_max_min_cell_probability": (
                        float(growth_veto_gate.max_min_cell_probability)
                        if growth_veto_gate.max_min_cell_probability is not None
                        else float("nan")
                    ),
                    "growth_veto_max_local_neighbor_distortion": (
                        float(growth_veto_gate.max_local_neighbor_distortion)
                        if growth_veto_gate.max_local_neighbor_distortion is not None
                        else float("nan")
                    ),
                    "growth_veto_min_anchor_count": int(
                        growth_veto_gate.min_anchor_count
                    ),
                    "growth_veto_min_complete_component_size": int(
                        growth_veto_gate.min_complete_component_size
                        if growth_veto_gate.min_complete_component_size is not None
                        else 0
                    ),
                    "growth_veto_max_row_rank": int(growth_veto_gate.max_row_rank),
                    "growth_veto_max_column_rank": int(
                        growth_veto_gate.max_column_rank
                    ),
                }
            )

    return GrowthVetoCleanupResult(
        tuple(results),
        tuple(diagnostic_rows),
        tuple(_summary_rows(diagnostic_rows)),
    )


def _method_for_prediction_base(prediction_base: GrowthVetoPredictionBase) -> str:
    if prediction_base == "coherence-suffix":
        return COHERENCE_SUFFIX_METHOD
    return METHOD


def _variant_for_prediction_base(prediction_base: GrowthVetoPredictionBase) -> str:
    if prediction_base == "coherence-suffix":
        return "CoherenceSuffixStitch + growth-veto cleanup"
    return "CoherenceSuffixTeacherRescue + growth-veto cleanup"


def growth_veto_gate_reason(
    row: Mapping[str, Any], gate: GrowthVetoGate, *, n_sessions: int
) -> str:
    """Return ``accepted`` or the first label-free rejection reason for a row."""

    if gate.require_not_suffix_edge and str(row.get("edge_source", "")) == "suffix":
        return "coherence_suffix_edge"
    if str(row.get("remove_reason", "")) != "split_edge":
        return "not_splittable"
    if int(row.get("would_split_component", 0)) <= 0:
        return "does_not_split_component"
    if gate.require_terminal_edge and int(row.get("is_terminal_edge", 0)) <= 0:
        return "not_terminal_edge"
    if gate.require_last_session_edge and int(row.get("is_last_session_edge", 0)) <= 0:
        return "not_last_session_edge"
    if gate.require_complete_component and int(
        row.get("complete_component_size", 0)
    ) < int(n_sessions):
        return "not_complete_component"
    if gate.min_complete_component_size is not None and int(
        row.get("complete_component_size", 0)
    ) < int(gate.min_complete_component_size):
        return "complete_component_size_below_gate"
    if int(row.get("growth_anchor_count", 0)) < int(gate.min_anchor_count):
        return "growth_anchor_count_below_gate"
    growth_residual = _finite_float(row.get("growth_residual"), float("nan"))
    if not np.isfinite(growth_residual) or growth_residual < float(
        gate.min_growth_residual
    ):
        return "growth_residual_below_gate"
    for key, threshold in (
        ("growth_residual_mahalanobis", gate.min_growth_residual_mahalanobis),
        ("registered_iou", gate.min_registered_iou),
        ("shifted_iou", gate.min_shifted_iou),
    ):
        value = _finite_float(row.get(key), float("nan"))
        if not np.isfinite(value) or value < float(threshold):
            return f"{key}_below_gate"
    for key, threshold in (
        ("registered_iou", gate.max_registered_iou),
        ("shifted_iou", gate.max_shifted_iou),
    ):
        if threshold is None:
            continue
        value = _finite_float(row.get(key), float("nan"))
        if not np.isfinite(value) or value > float(threshold):
            return f"{key}_above_gate"
    row_rank = int(_finite_float(row.get("row_rank"), float("inf")))
    column_rank = int(_finite_float(row.get("column_rank"), float("inf")))
    if row_rank <= 0 or row_rank > int(gate.max_row_rank):
        return "row_rank_above_gate"
    if column_rank <= 0 or column_rank > int(gate.max_column_rank):
        return "column_rank_above_gate"
    cell_a = _finite_float(row.get("cell_probability_a"), float("nan"))
    cell_b = _finite_float(row.get("cell_probability_b"), float("nan"))
    if not np.isfinite(cell_a) or not np.isfinite(cell_b):
        return "cell_probability_missing"
    if min(cell_a, cell_b) < float(gate.min_cell_probability):
        return "cell_probability_below_gate"
    if gate.max_min_cell_probability is not None and min(cell_a, cell_b) > float(
        gate.max_min_cell_probability
    ):
        return "min_cell_probability_above_gate"
    if gate.max_local_neighbor_distortion is not None:
        distortion = _finite_float(row.get("local_neighbor_distortion"), float("nan"))
        if not np.isfinite(distortion) or distortion > float(
            gate.max_local_neighbor_distortion
        ):
            return "local_neighbor_distortion_above_gate"
    return "accepted"


def _selected_growth_veto_rows(
    rows: Sequence[Mapping[str, Any]], *, gate: GrowthVetoGate, n_sessions: int
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if growth_veto_gate_reason(row, gate, n_sessions=n_sessions) == "accepted"
    ]
    selected.sort(key=_growth_veto_sort_key)
    return selected[: int(gate.max_vetoes_per_subject)]


def _augment_growth_veto_candidate_shifted_iou(
    rows: list[dict[str, Any]],
    sessions: Sequence[Any],
    *,
    gate: GrowthVetoGate,
    n_sessions: int,
) -> list[dict[str, Any]]:
    requested_by_pair: dict[tuple[int, int, str], list[tuple[int, int, int]]] = (
        defaultdict(list)
    )
    for row_index, row in enumerate(rows):
        if np.isfinite(_finite_float(row.get("shifted_iou"), float("nan"))):
            continue
        if not _needs_sparse_shifted_iou(row, gate, n_sessions=n_sessions):
            continue
        session_a = int(row["session_a"])
        session_b = int(row["session_b"])
        transform_type = str(row.get("transform_type", "affine"))
        requested_by_pair[(session_a, session_b, transform_type)].append(
            (row_index, int(row["roi_a"]), int(row["roi_b"]))
        )
    for (session_a, session_b, transform_type), requested in requested_by_pair.items():
        if session_a < 0 or session_b >= len(sessions):
            continue
        shifted_values = _sparse_shifted_iou_for_edges(
            sessions,
            session_a=session_a,
            session_b=session_b,
            transform_type=transform_type,
            requested_edges=[(roi_a, roi_b) for _index, roi_a, roi_b in requested],
        )
        for row_index, roi_a, roi_b in requested:
            value = shifted_values.get((int(roi_a), int(roi_b)), float("nan"))
            rows[int(row_index)]["shifted_iou"] = float(value)
    return rows


def _needs_sparse_shifted_iou(
    row: Mapping[str, Any], gate: GrowthVetoGate, *, n_sessions: int
) -> bool:
    if gate.require_not_suffix_edge and str(row.get("edge_source", "")) == "suffix":
        return False
    if str(row.get("remove_reason", "")) != "split_edge":
        return False
    if int(row.get("would_split_component", 0)) <= 0:
        return False
    if gate.require_terminal_edge and int(row.get("is_terminal_edge", 0)) <= 0:
        return False
    if gate.require_last_session_edge and int(row.get("is_last_session_edge", 0)) <= 0:
        return False
    if gate.require_complete_component and int(
        row.get("complete_component_size", 0)
    ) < int(n_sessions):
        return False
    if gate.min_complete_component_size is not None and int(
        row.get("complete_component_size", 0)
    ) < int(gate.min_complete_component_size):
        return False
    if int(row.get("growth_anchor_count", 0)) < int(gate.min_anchor_count):
        return False
    growth_residual = _finite_float(row.get("growth_residual"), float("nan"))
    if not np.isfinite(growth_residual) or growth_residual < float(
        gate.min_growth_residual
    ):
        return False
    for key, threshold in (
        ("growth_residual_mahalanobis", gate.min_growth_residual_mahalanobis),
        ("registered_iou", gate.min_registered_iou),
    ):
        value = _finite_float(row.get(key), float("nan"))
        if not np.isfinite(value) or value < float(threshold):
            return False
    if gate.max_registered_iou is not None:
        registered_iou = _finite_float(row.get("registered_iou"), float("nan"))
        if not np.isfinite(registered_iou) or registered_iou > float(
            gate.max_registered_iou
        ):
            return False
    row_rank = int(_finite_float(row.get("row_rank"), float("inf")))
    column_rank = int(_finite_float(row.get("column_rank"), float("inf")))
    if row_rank <= 0 or row_rank > int(gate.max_row_rank):
        return False
    if column_rank <= 0 or column_rank > int(gate.max_column_rank):
        return False
    cell_a = _finite_float(row.get("cell_probability_a"), float("nan"))
    cell_b = _finite_float(row.get("cell_probability_b"), float("nan"))
    if not np.isfinite(cell_a) or not np.isfinite(cell_b):
        return False
    min_cell_probability = min(cell_a, cell_b)
    if min_cell_probability < float(gate.min_cell_probability):
        return False
    if gate.max_min_cell_probability is not None:
        if min_cell_probability > float(gate.max_min_cell_probability):
            return False
    if gate.max_local_neighbor_distortion is not None:
        distortion = _finite_float(row.get("local_neighbor_distortion"), float("nan"))
        if not np.isfinite(distortion) or distortion > float(
            gate.max_local_neighbor_distortion
        ):
            return False
    return True


def _sparse_shifted_iou_for_edges(
    sessions: Sequence[Any],
    *,
    session_a: int,
    session_b: int,
    transform_type: str,
    requested_edges: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    source_indices = _roi_indices(sessions[int(session_a)])
    target_indices = _roi_indices(sessions[int(session_b)])
    source_local = {int(roi): index for index, roi in enumerate(source_indices)}
    target_local = {int(roi): index for index, roi in enumerate(target_indices)}
    source_locs = sorted(
        {
            int(source_local[int(roi_a)])
            for roi_a, _roi_b in requested_edges
            if int(roi_a) in source_local
        }
    )
    target_locs = sorted(
        {
            int(target_local[int(roi_b)])
            for _roi_a, roi_b in requested_edges
            if int(roi_b) in target_local
        }
    )
    if not source_locs or not target_locs:
        return {}
    registered = register_plane_pair(
        sessions[int(session_a)].plane_data,
        sessions[int(session_b)].plane_data,
        transform_type=str(transform_type),
    )
    reference_masks = (
        np.asarray(sessions[int(session_a)].plane_data.roi_masks)[source_locs] > 0
    )
    moving_masks = np.asarray(registered.roi_masks)[target_locs] > 0
    shifted = _pairwise_shifted_iou_from_support(
        reference_masks,
        moving_masks,
        radius=2,
    )["shifted_iou"]
    source_position = {
        int(source_indices[local]): pos for pos, local in enumerate(source_locs)
    }
    target_position = {
        int(target_indices[local]): pos for pos, local in enumerate(target_locs)
    }
    output: dict[tuple[int, int], float] = {}
    for roi_a, roi_b in requested_edges:
        source_pos = source_position.get(int(roi_a))
        target_pos = target_position.get(int(roi_b))
        if source_pos is None or target_pos is None:
            continue
        output[(int(roi_a), int(roi_b))] = float(shifted[source_pos, target_pos])
    return output


def _apply_growth_veto_rows(
    tracks: np.ndarray, rows: Sequence[Mapping[str, Any]], *, gate: GrowthVetoGate
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int, int], ...]]:
    output = veto._as_track_matrix(tracks)
    n_sessions = int(output.shape[1]) if output.ndim == 2 else 0
    applied: list[tuple[int, int, int, int, int]] = []
    for row in rows[: int(gate.max_vetoes_per_subject)]:
        edge = _edge_from_row(row)
        occurrence_index = int(row.get("occurrence_index", 0))
        split = veto._remove_edge_occurrence(
            output, edge, occurrence_index=occurrence_index
        )
        if split.reason != "split_edge" or int(split.would_split_component) <= 0:
            continue
        if not _split_satisfies_current_structural_gate(
            split,
            gate=gate,
            n_sessions=n_sessions,
        ):
            continue
        output = veto._as_track_matrix(split.tracks)
        applied.append((*edge, occurrence_index))
    return output, tuple(applied)


def _split_satisfies_current_structural_gate(
    split: Any, *, gate: GrowthVetoGate, n_sessions: int
) -> bool:
    if gate.require_terminal_edge and int(split.is_terminal_edge) <= 0:
        return False
    if gate.require_last_session_edge and int(split.is_last_session_edge) <= 0:
        return False
    if gate.require_complete_component and int(split.complete_component_size) < int(
        n_sessions
    ):
        return False
    if gate.min_complete_component_size is not None and int(
        split.complete_component_size
    ) < int(gate.min_complete_component_size):
        return False
    return True


def _growth_veto_sort_key(
    row: Mapping[str, Any],
) -> tuple[float, float, float, int, int, int]:
    return (
        -_finite_float(row.get("growth_residual_mahalanobis"), 0.0),
        -_finite_float(row.get("shifted_iou"), 0.0),
        -_finite_float(row.get("registered_iou"), 0.0),
        int(row.get("session_a", 0)),
        int(row.get("session_b", 0)),
        int(row.get("roi_a", 0)),
    )


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row.get("subject", ""))].append(row)
    by_subject["ALL"] = list(rows)
    output: list[dict[str, Any]] = []
    for subject, subject_rows in sorted(by_subject.items()):
        selected = [
            row for row in subject_rows if int(row.get("selected_by_growth_veto", 0))
        ]
        applied = [
            row for row in subject_rows if int(row.get("applied_by_growth_veto", 0))
        ]
        output.append(
            {
                "subject": subject,
                "accepted_edges": int(len(subject_rows)),
                "selected_by_growth_veto": int(len(selected)),
                "applied_by_growth_veto": int(len(applied)),
                "selected_true_positive_edges": int(
                    sum(
                        str(row.get("edge_status_against_gt")) == "true_positive"
                        for row in selected
                    )
                ),
                "selected_false_positive_edges": int(
                    sum(
                        str(row.get("edge_status_against_gt")) == "false_positive"
                        for row in selected
                    )
                ),
                "applied_true_positive_edges": int(
                    sum(
                        str(row.get("edge_status_against_gt")) == "true_positive"
                        for row in applied
                    )
                ),
                "applied_false_positive_edges": int(
                    sum(
                        str(row.get("edge_status_against_gt")) == "false_positive"
                        for row in applied
                    )
                ),
            }
        )
    return output


def _edge_from_row(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["session_a"]),
        int(row["session_b"]),
        int(row["roi_a"]),
        int(row["roi_b"]),
    )


def _edge_row_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    edge = _edge_from_row(row)
    return (*edge, int(row.get("occurrence_index", 0)))


def _finite_float(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return numeric if np.isfinite(numeric) else float(fallback)


def _optional_float_arg(value: str) -> float | None:
    if str(value).strip().lower() in {"none", "null", "off", "disabled"}:
        return None
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the growth-veto cleanup parser."""

    parser = veto.build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-policy-growth-veto-cleanup"
    parser.description = (
        "Run CoherenceSuffixTeacherRescue and split accepted adjacent edges that "
        "pass a strict label-free growth-veto gate."
    )
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--diagnostics-format", choices=("csv", "json"), default="csv")
    parser.add_argument(
        "--min-growth-residual-mahalanobis",
        "--growth-veto-min-mahalanobis",
        dest="min_growth_residual_mahalanobis",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--min-growth-residual",
        "--growth-veto-min-residual",
        dest="min_growth_residual",
        type=float,
        default=2.5,
        help=(
            "Require an absolute growth-field residual, in pixels, in addition "
            "to the Mahalanobis residual. This prevents vetoes caused only by an "
            "over-confident/tiny growth covariance on a visually small displacement."
        ),
    )
    parser.add_argument(
        "--min-veto-registered-iou",
        "--growth-veto-min-registered-iou",
        dest="min_veto_registered_iou",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--max-veto-registered-iou",
        "--growth-veto-max-registered-iou",
        dest="max_veto_registered_iou",
        type=_optional_float_arg,
        default=0.60,
        help=(
            "Optional upper bound on registered IoU for growth-veto candidates. "
            "This lets benchmark rows test the high-growth / moderate-local-evidence "
            "pocket while avoiding very strong local matches."
        ),
    )
    parser.add_argument(
        "--min-veto-shifted-iou",
        "--growth-veto-min-shifted-iou",
        dest="min_veto_shifted_iou",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--max-veto-shifted-iou",
        "--growth-veto-max-shifted-iou",
        dest="max_veto_shifted_iou",
        type=_optional_float_arg,
        default=0.80,
        help=(
            "Optional upper bound on shifted IoU for growth-veto candidates. "
            "Use with --growth-veto-max-registered-iou to avoid vetoing "
            "high-confidence terminal true continuations."
        ),
    )
    parser.add_argument(
        "--min-veto-cell-probability",
        "--growth-veto-min-cell-probability",
        dest="min_veto_cell_probability",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--max-veto-min-cell-probability",
        "--growth-veto-max-min-cell-probability",
        dest="max_veto_min_cell_probability",
        type=_optional_float_arg,
        default=0.65,
        help=(
            "Optional upper bound on min(cell_probability_a, cell_probability_b). "
            "This lets a growth veto target weak-endpoint continuations instead "
            "of high-confidence true terminal edges."
        ),
    )
    parser.add_argument(
        "--max-veto-local-neighbor-distortion",
        "--growth-veto-max-local-neighbor-distortion",
        dest="max_veto_local_neighbor_distortion",
        type=_optional_float_arg,
        default=0.05,
        help=(
            "Optional upper bound on the local neighbor-distance distortion for "
            "growth-veto candidates. This keeps the veto focused on edges that "
            "are globally growth-field outliers but still locally coherent. "
            "Use 'none' to disable this optional cap."
        ),
    )
    parser.add_argument(
        "--min-veto-anchor-count",
        "--growth-veto-min-anchor-count",
        dest="min_veto_anchor_count",
        type=suffix._nonnegative_int_arg,
        default=0,
        help=(
            "Require at least this many growth-field anchor edges for the "
            "session pair before applying a veto."
        ),
    )
    parser.add_argument(
        "--min-veto-complete-component-size",
        "--growth-veto-min-complete-component-size",
        dest="min_veto_complete_component_size",
        type=suffix._nonnegative_int_arg,
        default=None,
        help=(
            "Optional minimum complete-component size for veto candidates. "
            "This is stricter than --require-veto-complete-component when set "
            "above the number of sessions, and explicit enough for benchmark "
            "scripts that record the intended operating point."
        ),
    )
    parser.add_argument("--max-veto-row-rank", type=suffix._positive_int_arg, default=1)
    parser.add_argument(
        "--max-veto-column-rank", type=suffix._positive_int_arg, default=1
    )
    parser.add_argument(
        "--require-veto-not-suffix-edge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-veto-terminal-edge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-veto-last-session-edge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-veto-complete-component",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-vetoes-per-subject",
        "--growth-veto-max-vetoes-per-subject",
        dest="max_vetoes_per_subject",
        type=suffix._positive_int_arg,
        default=1,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the growth-veto cleanup benchmark."""

    args = build_arg_parser().parse_args(argv)
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        plane_name=args.plane_name,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        max_gap=args.max_gap,
        transform_type=args.transform_type,
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
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    suffix_gate = suffix.CoherenceSuffixStitchGate(
        suffix_path_length=int(args.suffix_path_length),
        min_cell_probability=float(args.min_cell_probability),
        min_area_ratio=float(args.min_area_ratio),
        max_centroid_distance=float(args.max_centroid_distance),
        min_shifted_iou=float(args.min_shifted_iou),
        min_motion_consistency=float(args.min_motion_consistency),
        min_shape_consistency=float(args.min_shape_consistency),
        max_stitches_per_subject=int(args.max_stitches_per_subject),
    )
    result = run_track2p_policy_growth_veto_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
        suffix_gate=suffix_gate,
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
        anchor_min_registered_iou=float(args.anchor_min_registered_iou),
        anchor_min_shifted_iou=float(args.anchor_min_shifted_iou),
        anchor_min_cell_probability=float(args.anchor_min_cell_probability),
        growth_veto_gate=GrowthVetoGate(
            min_growth_residual_mahalanobis=float(args.min_growth_residual_mahalanobis),
            min_growth_residual=float(args.min_growth_residual),
            min_registered_iou=float(args.min_veto_registered_iou),
            min_shifted_iou=float(args.min_veto_shifted_iou),
            max_registered_iou=args.max_veto_registered_iou,
            max_shifted_iou=args.max_veto_shifted_iou,
            min_cell_probability=float(args.min_veto_cell_probability),
            max_min_cell_probability=(
                None
                if args.max_veto_min_cell_probability is None
                else float(args.max_veto_min_cell_probability)
            ),
            max_local_neighbor_distortion=(
                None
                if args.max_veto_local_neighbor_distortion is None
                else float(args.max_veto_local_neighbor_distortion)
            ),
            min_anchor_count=int(args.min_veto_anchor_count),
            min_complete_component_size=(
                None
                if args.min_veto_complete_component_size is None
                else int(args.min_veto_complete_component_size)
            ),
            max_row_rank=int(args.max_veto_row_rank),
            max_column_rank=int(args.max_veto_column_rank),
            require_not_suffix_edge=bool(args.require_veto_not_suffix_edge),
            require_terminal_edge=bool(args.require_veto_terminal_edge),
            require_last_session_edge=bool(args.require_veto_last_session_edge),
            require_complete_component=bool(args.require_veto_complete_component),
            max_vetoes_per_subject=int(args.max_vetoes_per_subject),
        ),
        prediction_base=cast(GrowthVetoPredictionBase, args.growth_veto_base),
        progress=bool(args.progress),
    )
    rows = [benchmark_result.to_dict() for benchmark_result in result.results]
    write_results(rows, args.output, cast(OutputFormat, args.format))
    if args.diagnostics_output is not None:
        veto.write_rows(
            result.edge_rows,
            args.diagnostics_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    if args.summary_output is not None:
        veto.write_rows(
            result.summary_rows,
            args.summary_output,
            output_format=cast(Literal["csv", "json"], args.format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
