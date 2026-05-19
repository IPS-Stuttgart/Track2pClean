"""Cause-specific rejection ledger for global Track2p assignment errors."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import (
    GlobalAssignmentRun,
    SessionEdge,
)
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.reference import Track2pReference

LedgerValue = float | int | str

REJECTION_REASONS = (
    "selected_by_solver",
    "reference_roi_missing_from_loaded_session",
    "measurement_roi_missing_from_loaded_session",
    "both_rois_missing_from_loaded_sessions",
    "true_edge_outside_cost_matrix",
    "true_edge_nonfinite_cost",
    "true_edge_large_cost_or_empty_registered_roi",
    "true_edge_gated_by_cost_threshold",
    "true_edge_not_row_top_k",
    "true_edge_not_column_top_k",
    "wrong_edge_selected",
    "mutual_top1_rejected_by_solver_prior",
    "unclassified_solver_rejection",
)

LEDGER_FIELDNAMES = [
    "subject",
    "session_a",
    "session_b",
    "session_a_name",
    "session_b_name",
    "session_gap",
    "reference_roi_index",
    "measurement_roi_index",
    "reference_loaded_index",
    "measurement_loaded_index",
    "true_cost",
    "adjusted_true_cost",
    "true_is_finite",
    "row_rank",
    "column_rank",
    "row_top_k",
    "column_top_k",
    "mutual_top1",
    "selected_by_solver",
    "selected_row_target_roi_index",
    "selected_column_source_roi_index",
    "gated_by_threshold",
    "large_cost_like",
    "rejection_reason",
]


@dataclass(frozen=True)
class SolverRejectionLedger:
    """Detailed per-GT-edge rows plus subject-level summary fields."""

    rows: tuple[dict[str, LedgerValue], ...]
    summary: dict[str, float | int]


def build_solver_rejection_ledger(
    assignment: GlobalAssignmentRun,
    sessions: Sequence[Track2pSession],
    reference: Track2pReference,
    *,
    subject: str = "",
    curated_only: bool = False,
    cost_threshold: float | None = 6.0,
    gap_penalty: float = 1.0,
    rank_k: int = 1,
    large_cost: float = 1.0e5,
) -> SolverRejectionLedger:
    """Classify why each manual-GT pairwise edge was accepted or rejected.

    The ledger works in the solver's loaded-ROI coordinate system but reports
    Suite2p ROI indices in its public rows.  This makes it useful for diagnosing
    whether complete-track F1 failures are caused by ROI filtering/indexing,
    pairwise ranking, solver cost gating, or path-cover priors.
    """

    sessions = tuple(sessions)
    if len(sessions) != reference.n_sessions:
        raise ValueError(
            "sessions and reference must have the same number of sessions"
        )
    if rank_k < 1:
        raise ValueError("rank_k must be at least one")
    if large_cost < 0.0:
        raise ValueError("large_cost must be non-negative")

    suite2p_to_loaded = tuple(
        _loaded_index_by_suite2p_index(session) for session in sessions
    )
    loaded_to_suite2p = tuple(
        _suite2p_indices_by_loaded_index(session) for session in sessions
    )
    selected = _selected_solver_edges_by_session_edge(assignment)

    rows: list[dict[str, LedgerValue]] = []
    for edge in assignment.session_edges:
        session_a, session_b = int(edge[0]), int(edge[1])
        cost_matrix = np.asarray(assignment.pairwise_costs[edge], dtype=float)
        selected_for_edge = selected.get(edge, _empty_selected_edge_maps())
        reference_matches = reference.pairwise_matches(
            session_a, session_b, curated_only=curated_only
        )
        for reference_roi, measurement_roi in reference_matches:
            rows.append(
                _ledger_row_for_reference_edge(
                    subject=subject,
                    session_a=session_a,
                    session_b=session_b,
                    session_a_name=_session_name(sessions[session_a], session_a),
                    session_b_name=_session_name(sessions[session_b], session_b),
                    reference_roi=int(reference_roi),
                    measurement_roi=int(measurement_roi),
                    reference_loaded=suite2p_to_loaded[session_a].get(
                        int(reference_roi)
                    ),
                    measurement_loaded=suite2p_to_loaded[session_b].get(
                        int(measurement_roi)
                    ),
                    cost_matrix=cost_matrix,
                    selected_for_edge=selected_for_edge,
                    target_suite2p_by_loaded=loaded_to_suite2p[session_b],
                    source_suite2p_by_loaded=loaded_to_suite2p[session_a],
                    cost_threshold=cost_threshold,
                    gap_penalty=gap_penalty,
                    rank_k=rank_k,
                    large_cost=large_cost,
                )
            )

    return SolverRejectionLedger(rows=tuple(rows), summary=summarize_ledger_rows(rows))


def summarize_ledger_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Return flat benchmark-score fields from detailed ledger rows."""

    rows = tuple(rows)
    counts = Counter(str(row.get("rejection_reason", "")) for row in rows)
    selected = int(counts.get("selected_by_solver", 0))
    total = int(len(rows))
    rejected = int(total - selected)
    summary: dict[str, float | int] = {
        "solver_ledger_gt_edges": total,
        "solver_ledger_selected_edges": selected,
        "solver_ledger_rejected_edges": rejected,
        "solver_ledger_selected_rate": _rate(selected, total),
        "solver_ledger_rejected_rate": _rate(rejected, total),
        "solver_ledger_median_row_rank": _median_positive_int(rows, "row_rank"),
        "solver_ledger_median_column_rank": _median_positive_int(
            rows, "column_rank"
        ),
    }
    for reason in REJECTION_REASONS:
        summary[f"solver_ledger_{reason}"] = int(counts.get(reason, 0))
    return summary


def write_solver_rejection_ledger_rows(
    rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    """Write detailed solver-rejection rows as CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ledger_fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ledger_row_for_reference_edge(
    *,
    subject: str,
    session_a: int,
    session_b: int,
    session_a_name: str,
    session_b_name: str,
    reference_roi: int,
    measurement_roi: int,
    reference_loaded: int | None,
    measurement_loaded: int | None,
    cost_matrix: np.ndarray,
    selected_for_edge: Mapping[str, Mapping[Any, Any]],
    target_suite2p_by_loaded: np.ndarray,
    source_suite2p_by_loaded: np.ndarray,
    cost_threshold: float | None,
    gap_penalty: float,
    rank_k: int,
    large_cost: float,
) -> dict[str, LedgerValue]:
    session_gap = int(session_b - session_a)
    base_row: dict[str, LedgerValue] = {
        "subject": subject,
        "session_a": int(session_a),
        "session_b": int(session_b),
        "session_a_name": session_a_name,
        "session_b_name": session_b_name,
        "session_gap": session_gap,
        "reference_roi_index": int(reference_roi),
        "measurement_roi_index": int(measurement_roi),
        "reference_loaded_index": -1 if reference_loaded is None else reference_loaded,
        "measurement_loaded_index": -1
        if measurement_loaded is None
        else measurement_loaded,
        "true_cost": np.nan,
        "adjusted_true_cost": np.nan,
        "true_is_finite": 0,
        "row_rank": -1,
        "column_rank": -1,
        "row_top_k": 0,
        "column_top_k": 0,
        "mutual_top1": 0,
        "selected_by_solver": 0,
        "selected_row_target_roi_index": -1,
        "selected_column_source_roi_index": -1,
        "gated_by_threshold": 0,
        "large_cost_like": 0,
    }

    missing_reason = _loaded_index_missing_reason(reference_loaded, measurement_loaded)
    if missing_reason:
        return {**base_row, "rejection_reason": missing_reason}
    assert reference_loaded is not None
    assert measurement_loaded is not None

    if not _inside_cost_matrix(cost_matrix, reference_loaded, measurement_loaded):
        return {**base_row, "rejection_reason": "true_edge_outside_cost_matrix"}

    true_cost = float(cost_matrix[reference_loaded, measurement_loaded])
    true_is_finite = bool(np.isfinite(true_cost))
    adjusted_true_cost = (
        true_cost + float(gap_penalty) * max(0, session_gap - 1)
        if true_is_finite
        else np.nan
    )
    row_rank = _cost_rank(cost_matrix[reference_loaded, :], measurement_loaded)
    column_rank = _cost_rank(cost_matrix[:, measurement_loaded], reference_loaded)
    row_top_k = 0 < row_rank <= rank_k
    column_top_k = 0 < column_rank <= rank_k
    mutual_top1 = row_rank == 1 and column_rank == 1
    selected_pairs = selected_for_edge["pairs"]
    row_to_column = selected_for_edge["row_to_column"]
    column_to_row = selected_for_edge["column_to_row"]
    selected_by_solver = (reference_loaded, measurement_loaded) in selected_pairs
    selected_row_target = row_to_column.get(reference_loaded)
    selected_column_source = column_to_row.get(measurement_loaded)
    gated = (
        cost_threshold is not None
        and true_is_finite
        and adjusted_true_cost > float(cost_threshold)
    )
    large_cost_like = true_is_finite and true_cost >= float(large_cost)

    row = {
        **base_row,
        "true_cost": true_cost,
        "adjusted_true_cost": float(adjusted_true_cost)
        if np.isfinite(adjusted_true_cost)
        else np.nan,
        "true_is_finite": int(true_is_finite),
        "row_rank": int(row_rank),
        "column_rank": int(column_rank),
        "row_top_k": int(row_top_k),
        "column_top_k": int(column_top_k),
        "mutual_top1": int(mutual_top1),
        "selected_by_solver": int(selected_by_solver),
        "selected_row_target_roi_index": _suite2p_or_minus_one(
            target_suite2p_by_loaded, selected_row_target
        ),
        "selected_column_source_roi_index": _suite2p_or_minus_one(
            source_suite2p_by_loaded, selected_column_source
        ),
        "gated_by_threshold": int(gated),
        "large_cost_like": int(large_cost_like),
    }
    return {**row, "rejection_reason": _rejection_reason(row)}


def _rejection_reason(row: Mapping[str, Any]) -> str:
    if int(row["selected_by_solver"]):
        return "selected_by_solver"
    if not int(row["true_is_finite"]):
        return "true_edge_nonfinite_cost"
    if int(row["large_cost_like"]):
        return "true_edge_large_cost_or_empty_registered_roi"
    if int(row["gated_by_threshold"]):
        return "true_edge_gated_by_cost_threshold"
    if not int(row["row_top_k"]):
        return "true_edge_not_row_top_k"
    if not int(row["column_top_k"]):
        return "true_edge_not_column_top_k"
    if (
        int(row["selected_row_target_roi_index"]) >= 0
        or int(row["selected_column_source_roi_index"]) >= 0
    ):
        return "wrong_edge_selected"
    if int(row["mutual_top1"]):
        return "mutual_top1_rejected_by_solver_prior"
    return "unclassified_solver_rejection"


def _selected_solver_edges_by_session_edge(
    assignment: GlobalAssignmentRun,
) -> dict[SessionEdge, dict[str, dict[Any, Any]]]:
    selected = {edge: _empty_selected_edge_maps() for edge in assignment.session_edges}
    tracks = getattr(assignment.result, "tracks", ())
    for track_index, track in enumerate(tracks):
        normalized_track = {int(session): int(roi) for session, roi in track.items()}
        for edge in assignment.session_edges:
            session_a, session_b = int(edge[0]), int(edge[1])
            if session_a not in normalized_track or session_b not in normalized_track:
                continue
            reference_loaded = normalized_track[session_a]
            measurement_loaded = normalized_track[session_b]
            selected[edge]["pairs"][(reference_loaded, measurement_loaded)] = int(
                track_index
            )
            selected[edge]["row_to_column"].setdefault(
                reference_loaded, measurement_loaded
            )
            selected[edge]["column_to_row"].setdefault(
                measurement_loaded, reference_loaded
            )
    return selected


def _empty_selected_edge_maps() -> dict[str, dict[Any, Any]]:
    return {"pairs": {}, "row_to_column": {}, "column_to_row": {}}


def _loaded_index_by_suite2p_index(session: Track2pSession) -> dict[int, int]:
    return {
        int(suite2p_index): int(loaded_index)
        for loaded_index, suite2p_index in enumerate(
            _suite2p_indices_by_loaded_index(session)
        )
    }


def _suite2p_indices_by_loaded_index(session: Track2pSession) -> np.ndarray:
    plane = session.plane_data
    if plane.roi_indices is None:
        return np.arange(int(plane.n_rois), dtype=int)
    return np.asarray(plane.roi_indices, dtype=int).reshape(-1)


def _session_name(session: Track2pSession, session_index: int) -> str:
    return str(getattr(session, "session_name", session_index))


def _loaded_index_missing_reason(
    reference_loaded: int | None, measurement_loaded: int | None
) -> str:
    if reference_loaded is not None and measurement_loaded is not None:
        return ""
    if reference_loaded is None and measurement_loaded is None:
        return "both_rois_missing_from_loaded_sessions"
    if reference_loaded is None:
        return "reference_roi_missing_from_loaded_session"
    return "measurement_roi_missing_from_loaded_session"


def _inside_cost_matrix(
    cost_matrix: np.ndarray, reference_loaded: int, measurement_loaded: int
) -> bool:
    return (
        0 <= reference_loaded < cost_matrix.shape[0]
        and 0 <= measurement_loaded < cost_matrix.shape[1]
    )


def _cost_rank(values: np.ndarray, true_index: int) -> int:
    values = np.asarray(values, dtype=float).reshape(-1)
    if true_index < 0 or true_index >= values.size:
        return -1
    true_value = float(values[true_index])
    finite = np.isfinite(values)
    if not np.isfinite(true_value):
        return -1
    return int(1 + np.count_nonzero(finite & (values < true_value)))


def _suite2p_or_minus_one(
    suite2p_by_loaded: np.ndarray, loaded_index: Any | None
) -> int:
    if loaded_index is None:
        return -1
    loaded_index = int(loaded_index)
    if loaded_index < 0 or loaded_index >= suite2p_by_loaded.shape[0]:
        return -1
    return int(suite2p_by_loaded[loaded_index])


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count / total)


def _median_positive_int(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [int(row[key]) for row in rows if int(row.get(key, -1)) > 0]
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=float)))


def _ledger_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    extra = sorted({str(key) for row in rows for key in row} - set(LEDGER_FIELDNAMES))
    return LEDGER_FIELDNAMES + extra
