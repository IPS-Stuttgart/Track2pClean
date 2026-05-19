from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import GlobalAssignmentRun
from bayescatrack.evaluation.solver_rejection_ledger import (
    build_solver_rejection_ledger,
    write_solver_rejection_ledger_rows,
)
from bayescatrack.reference import Track2pReference


def test_solver_rejection_ledger_classifies_selected_ranking_gating_and_missing(
    tmp_path,
):
    sessions = (
        _session("s0", [10, 11, 12]),
        _session("s1", [20, 21, 22]),
    )
    reference = Track2pReference(
        session_names=("s0", "s1"),
        suite2p_indices=np.asarray(
            [[10, 20], [11, 21], [12, 22], [13, 23]], dtype=object
        ),
        source="manual_gt_test",
    )
    costs = np.asarray(
        [
            [0.1, 5.0, 9.0],
            [0.2, 0.3, 4.0],
            [9.0, 8.0, 7.0],
        ],
        dtype=float,
    )
    assignment = GlobalAssignmentRun(
        result=SimpleNamespace(tracks=[{0: 0, 1: 0}, {0: 1, 1: 0}]),
        pairwise_costs={(0, 1): costs},
        session_sizes=(3, 3),
        session_edges=((0, 1),),
    )

    ledger = build_solver_rejection_ledger(
        assignment,
        sessions,
        reference,
        subject="jm001",
        cost_threshold=6.0,
        rank_k=1,
    )

    reasons = {
        (row["reference_roi_index"], row["measurement_roi_index"]): row[
            "rejection_reason"
        ]
        for row in ledger.rows
    }
    assert reasons[(10, 20)] == "selected_by_solver"
    assert reasons[(11, 21)] == "true_edge_not_row_top_k"
    assert reasons[(12, 22)] == "true_edge_gated_by_cost_threshold"
    assert reasons[(13, 23)] == "both_rois_missing_from_loaded_sessions"

    assert ledger.summary["solver_ledger_gt_edges"] == 4
    assert ledger.summary["solver_ledger_selected_edges"] == 1
    assert ledger.summary["solver_ledger_rejected_edges"] == 3
    assert ledger.summary["solver_ledger_true_edge_not_row_top_k"] == 1
    assert ledger.summary["solver_ledger_true_edge_gated_by_cost_threshold"] == 1

    output_path = tmp_path / "ledger.csv"
    write_solver_rejection_ledger_rows(ledger.rows, output_path)
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["subject"] == "jm001"
    assert rows[0]["selected_by_solver"] == "1"


def test_solver_rejection_ledger_flags_mutual_top1_solver_prior_rejection():
    sessions = (_session("s0", [0]), _session("s1", [0]))
    reference = Track2pReference(
        session_names=("s0", "s1"),
        suite2p_indices=np.asarray([[0, 0]], dtype=object),
    )
    assignment = GlobalAssignmentRun(
        result=SimpleNamespace(tracks=[]),
        pairwise_costs={(0, 1): np.asarray([[0.1]], dtype=float)},
        session_sizes=(1, 1),
        session_edges=((0, 1),),
    )

    ledger = build_solver_rejection_ledger(assignment, sessions, reference)

    assert ledger.rows[0]["rejection_reason"] == "mutual_top1_rejected_by_solver_prior"
    assert ledger.summary["solver_ledger_mutual_top1_rejected_by_solver_prior"] == 1


def _session(name: str, roi_indices: list[int]):
    return SimpleNamespace(
        session_name=name,
        plane_data=SimpleNamespace(
            n_rois=len(roi_indices), roi_indices=np.asarray(roi_indices, dtype=int)
        ),
    )
