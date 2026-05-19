from __future__ import annotations

import pytest

from bayescatrack.experiments.track2p_solver_oracle_benchmark import (
    _expanded_variants,
    _validate_oracle_choices,
    format_solver_oracle_markdown,
    summarize_solver_oracle_rows,
)


def test_solver_oracle_summary_and_markdown() -> None:
    rows = [
        {
            "subject": "jm038",
            "variant": "Oracle edge costs + global assignment",
            "oracle": "edge-costs",
            "rank_k": "",
            "base_cost": "registered-iou",
            "oracle_registration_cost": "",
            "status": "ok",
            "pairwise_f1": 1.0,
            "complete_track_f1": 0.9,
            "pairwise_precision": 1.0,
            "pairwise_recall": 1.0,
            "complete_tracks": 11,
        },
        {
            "subject": "jm039",
            "variant": "Oracle edge costs + global assignment",
            "oracle": "edge-costs",
            "rank_k": "",
            "base_cost": "registered-iou",
            "oracle_registration_cost": "",
            "status": "ok",
            "pairwise_f1": 0.8,
            "complete_track_f1": 0.7,
            "pairwise_precision": 0.75,
            "pairwise_recall": 0.9,
            "complete_tracks": 7,
        },
        {
            "subject": "jm046",
            "variant": "Oracle edge costs + global assignment",
            "oracle": "edge-costs",
            "rank_k": "",
            "base_cost": "registered-iou",
            "oracle_registration_cost": "",
            "status": "failed",
            "error": "synthetic failure",
        },
    ]

    summary = summarize_solver_oracle_rows(rows)

    assert len(summary) == 1
    assert summary[0]["ok_subjects"] == 2
    assert summary[0]["failed_subjects"] == 1
    assert summary[0]["mean_pairwise_f1"] == pytest.approx(0.9)
    assert summary[0]["min_complete_track_f1"] == pytest.approx(0.7)

    markdown = format_solver_oracle_markdown(summary)
    assert "# Solver-oracle Track2p diagnostics" in markdown
    assert "Oracle edge costs + global assignment" in markdown
    assert "upper bounds/debugging checks" in markdown


def test_expanded_variants_expands_rank_k_only() -> None:
    assert _expanded_variants(("edge-costs", "rank-k"), (1, 3)) == (
        ("edge-costs", None),
        ("rank-k", 1),
        ("rank-k", 3),
    )


def test_solver_oracle_choice_validation_rejects_bad_rank_k() -> None:
    with pytest.raises(ValueError, match="rank-k values"):
        _validate_oracle_choices(("rank-k",), (0,), "registered-iou")
