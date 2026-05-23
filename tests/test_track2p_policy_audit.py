from collections import Counter

import numpy as np
from bayescatrack.experiments.track2p_policy_audit import (
    policy_edge_ledger_rows,
    track_edge_counter,
)


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
