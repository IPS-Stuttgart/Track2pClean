from __future__ import annotations

from bayescatrack.experiments import full_mht_scan_history_conflict_demo as demo


def test_scan_history_conflict_demo_rejects_local_motion_break() -> None:
    results = {result.policy: result for result in demo.evaluate_scan_history_conflict_demo()}

    assert results["local_score_only"].selected_candidate == "locally_high_score_motion_break"
    assert results["scan_history_pruning"].selected_candidate == "lower_score_motion_coherent"
    assert results["local_score_only"].path == (1, 20, 41)
    assert results["scan_history_pruning"].path == (1, 20, 30)
    assert results["local_score_only"].raw_score > results["scan_history_pruning"].raw_score
    assert results["scan_history_pruning"].pruning_score > results["local_score_only"].pruning_score
    assert results["local_score_only"].history_risk > 5.0
    assert results["scan_history_pruning"].history_risk == 0.0


def test_scan_history_conflict_demo_zero_weight_matches_local_score() -> None:
    results = {
        result.policy: result
        for result in demo.evaluate_scan_history_conflict_demo(
            scan_motion_history_weight=0.0,
        )
    }

    assert results["local_score_only"].selected_candidate == "locally_high_score_motion_break"
    assert results["scan_history_pruning"].selected_candidate == "locally_high_score_motion_break"
    assert results["scan_history_pruning"].pruning_score == results["scan_history_pruning"].raw_score


def test_scan_history_conflict_demo_candidate_rows_are_stable() -> None:
    rows = demo.candidate_rows()

    assert [row["candidate"] for row in rows] == [
        "locally_high_score_motion_break",
        "lower_score_motion_coherent",
    ]
    assert rows[0]["path"] == "1->20->41"
    assert rows[1]["path"] == "1->20->30"
    assert rows[0]["raw_score"] > rows[1]["raw_score"]
    assert rows[0]["pruning_score"] < rows[1]["pruning_score"]
