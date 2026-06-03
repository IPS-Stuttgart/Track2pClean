from __future__ import annotations

import csv
from pathlib import Path

from bayescatrack.experiments.track2p_component_residual_whatif import (
    OfficialCounts,
    discover_candidates,
    load_baseline_counts,
    score_candidates,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_baseline_counts_sums_subject_rows(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    _write_csv(
        baseline_path,
        [
            {
                "pairwise_true_positives": 10,
                "pairwise_false_positives": 2,
                "pairwise_false_negatives": 3,
                "complete_track_true_positives": 4,
                "complete_track_false_positives": 1,
                "complete_track_false_negatives": 1,
            },
            {
                "pairwise_true_positives": 5,
                "pairwise_false_positives": 1,
                "pairwise_false_negatives": 2,
                "complete_track_true_positives": 2,
                "complete_track_false_positives": 0,
                "complete_track_false_negatives": 1,
            },
        ],
    )

    counts = load_baseline_counts(baseline_path)

    assert counts == OfficialCounts(
        pairwise_tp=15,
        pairwise_fp=3,
        pairwise_fn=5,
        complete_tp=6,
        complete_fp=1,
        complete_fn=2,
    )


def test_discover_candidates_prefers_complete_track_repairs() -> None:
    residual_rows = [
        {
            "subject": "jm038",
            "error_type": "pairwise_fn",
            "session_a": "0",
            "session_b": "1",
            "roi_a": "10",
            "roi_b": "11",
            "track2p_supported": "1",
            "component_cleanup_supported": "0",
        },
        {
            "subject": "jm038",
            "error_type": "pairwise_fp",
            "session_a": "1",
            "session_b": "2",
            "roi_a": "20",
            "roi_b": "21",
            "track2p_supported": "0",
        },
        {
            "subject": "jm046",
            "error_type": "complete_track_fn",
            "reason_bucket": "missing seed-session ROI",
            "track2p_supported": "0",
        },
    ]
    baseline = OfficialCounts(
        pairwise_tp=586,
        pairwise_fp=26,
        pairwise_fn=19,
        complete_tp=56,
        complete_fp=3,
        complete_fn=4,
    )

    scored = score_candidates(discover_candidates(residual_rows), baseline)

    assert [row["edit_type"] for row in scored] == [
        "missing_seed_complete_fn_recovery",
        "track2p_supported_adjacent_fn_rescue",
        "bayes_only_pairwise_fp_veto",
    ]
    assert scored[0]["new_complete_track_f1"] > baseline.complete_track_f1
    assert scored[1]["new_pairwise_f1"] > baseline.pairwise_f1
