from collections import Counter

import numpy as np
from bayescatrack.experiments.track2p_policy_audit import (
    pairwise_edge_counter,
    pairwise_edge_ledger_rows,
    policy_edge_ledger_rows,
    track_edge_counter,
)


def _find_edge(rows, *, session_a, session_b, source_roi, target_roi):
    for row in rows:
        if (
            row["session_a"] == session_a
            and row["session_b"] == session_b
            and row["source_roi"] == source_roi
            and row["target_roi"] == target_roi
        ):
            return row
    raise AssertionError("edge not found")


def test_track_edge_counter_uses_consecutive_valid_observations() -> None:
    matrix = np.asarray(
        [
            [1, 2, 3],
            [4, -1, 5],
            [6, 7, 8],
        ],
        dtype=object,
    )

    assert track_edge_counter(matrix) == Counter(
        {
            (0, 1, 1, 2): 1,
            (1, 2, 2, 3): 1,
            (0, 1, 6, 7): 1,
            (1, 2, 7, 8): 1,
        }
    )


def test_policy_edge_ledger_is_duplicate_aware() -> None:
    predicted = np.asarray(
        [
            [1, 2, 3],
            [1, 2, 4],
            [5, 9, -1],
        ],
        dtype=object,
    )
    reference = np.asarray(
        [
            [1, 2, 3],
            [5, 7, 8],
        ],
        dtype=object,
    )

    rows = policy_edge_ledger_rows(
        predicted,
        reference,
        subject="mouse",
        session_names=("s0", "s1", "s2"),
        metadata={"threshold_method": "min"},
    )
    statuses = Counter(row["edge_status"] for row in rows)

    assert statuses == Counter(
        {"true_positive": 2, "false_positive": 3, "false_negative": 2}
    )
    assert all(row["subject"] == "mouse" for row in rows)
    assert {row["threshold_method"] for row in rows} == {"min"}


def test_pairwise_edge_counter_counts_duplicate_links() -> None:
    matrix = np.asarray(
        [
            [1, 2, 3],
            [1, 2, 4],
            [5, 6, -1],
        ],
        dtype=object,
    )

    counter = pairwise_edge_counter(matrix)

    assert counter[(0, 1, 1, 2)] == 2
    assert counter[(1, 2, 2, 3)] == 1
    assert counter[(1, 2, 2, 4)] == 1
    assert counter[(0, 1, 5, 6)] == 1
    assert (1, 2, 6, -1) not in counter


def test_pairwise_edge_ledger_rows_marks_tp_fp_fn_and_mixed_counts() -> None:
    predicted = np.asarray(
        [
            [1, 2, 3],
            [1, 2, 4],
            [5, 6, -1],
        ],
        dtype=object,
    )
    reference = np.asarray(
        [
            [1, 2, 3],
            [1, 2, 3],
            [7, 8, 9],
        ],
        dtype=object,
    )

    rows = pairwise_edge_ledger_rows(
        predicted,
        reference,
        subject="subject-a",
        session_names=("s0", "s1", "s2"),
    )

    duplicate_tp = _find_edge(
        rows, session_a=0, session_b=1, source_roi=1, target_roi=2
    )
    assert duplicate_tp["subject"] == "subject-a"
    assert duplicate_tp["session_a_name"] == "s0"
    assert duplicate_tp["session_b_name"] == "s1"
    assert duplicate_tp["predicted_count"] == 2
    assert duplicate_tp["reference_count"] == 2
    assert duplicate_tp["true_positive_count"] == 2
    assert duplicate_tp["classification"] == "true_positive"

    mixed = _find_edge(rows, session_a=1, session_b=2, source_roi=2, target_roi=3)
    assert mixed["predicted_count"] == 1
    assert mixed["reference_count"] == 2
    assert mixed["true_positive_count"] == 1
    assert mixed["false_negative_count"] == 1
    assert mixed["classification"] == "true_positive+false_negative"

    false_positive = _find_edge(
        rows, session_a=1, session_b=2, source_roi=2, target_roi=4
    )
    assert false_positive["false_positive_count"] == 1
    assert false_positive["classification"] == "false_positive"

    false_negative = _find_edge(
        rows, session_a=0, session_b=1, source_roi=7, target_roi=8
    )
    assert false_negative["false_negative_count"] == 1
    assert false_negative["classification"] == "false_negative"
