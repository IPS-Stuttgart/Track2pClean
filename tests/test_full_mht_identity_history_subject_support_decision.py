from __future__ import annotations

import csv

from bayescatrack.experiments.full_mht_identity_history_subject_support_decision import (
    evaluate_subject_support,
    format_subject_support_markdown,
    load_labeled_subject_rows,
    parse_labeled_input,
)


def _row(approach: str, subject: str, pairwise: float, complete: float) -> dict[str, str]:
    return {
        "approach": approach,
        "subject": subject,
        "pairwise_f1": str(pairwise),
        "complete_track_f1": str(complete),
    }


def _subject_rows(
    *,
    candidate_jm038_complete: float = 0.94,
    candidate_jm046_complete: float = 0.95,
    candidate_jm046_pairwise: float = 0.966,
    candidate_jm039_complete: float = 0.93,
    greedy_complete: float = 0.93,
    control_complete: float = 0.92,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for subject, candidate_complete, candidate_pairwise in (
        ("jm038", candidate_jm038_complete, 0.966),
        ("jm046", candidate_jm046_complete, candidate_jm046_pairwise),
        ("jm039", candidate_jm039_complete, 0.966),
    ):
        rows.extend(
            [
                _row("Track2p", subject, 0.960, control_complete),
                _row("FullMHTPrior2", subject, 0.960, control_complete),
                _row("FullMHTPriorSurvival", subject, 0.960, control_complete),
                _row("FullMHTNoPriorContinuation100", subject, 0.960, control_complete),
                _row("FullMHTIdentityHistoryNoLocalContext", subject, 0.960, control_complete),
                _row("FullMHTIdentityHistory", subject, candidate_pairwise, candidate_complete),
                _row("FullMHTGreedyIdentityHistory", subject, 0.966, greedy_complete),
            ]
        )
    return rows


def test_subject_support_accepts_multi_subject_complete_gain() -> None:
    decision = evaluate_subject_support(_subject_rows())

    assert decision["status"] == "complete"
    assert decision["subject_support_result"] == "stable_subject_support"
    assert decision["n_complete_gain_subjects"] == 2
    assert decision["complete_gain_subjects"] == ["jm038", "jm046"]
    assert decision["regression_subjects"] == []
    assert decision["worst_candidate_minus_greedy_complete_track_f1"] == 0.0


def test_subject_support_reports_missing_frozen_rows() -> None:
    rows = [
        row
        for row in _subject_rows()
        if not (row["subject"] == "jm046" and row["approach"] == "FullMHTPriorSurvival")
    ]

    decision = evaluate_subject_support(rows)

    assert decision["status"] == "incomplete"
    assert decision["subject_support_result"] == "missing_subject_rows"
    assert decision["missing_subject_approaches"] == {
        "jm046": ["FullMHTPriorSurvival"]
    }


def test_subject_support_rejects_single_subject_spike() -> None:
    decision = evaluate_subject_support(
        _subject_rows(candidate_jm046_complete=0.93, candidate_jm039_complete=0.93)
    )

    assert decision["status"] == "complete"
    assert decision["subject_support_result"] == "weak_subject_support"
    assert decision["complete_gain_subjects"] == ["jm038"]


def test_subject_support_rejects_subject_regression_vs_greedy() -> None:
    decision = evaluate_subject_support(
        _subject_rows(candidate_jm046_pairwise=0.950)
    )

    assert decision["subject_support_result"] == "subject_metric_regression"
    assert decision["greedy_regression_subjects"] == ["jm046"]
    assert decision["regression_subjects"] == ["jm046"]


def test_subject_support_rejects_subject_regression_vs_control() -> None:
    decision = evaluate_subject_support(
        _subject_rows(candidate_jm046_complete=0.95, control_complete=0.96)
    )

    assert decision["subject_support_result"] == "subject_metric_regression"
    assert sorted(decision["control_regression_subjects"]) == ["jm038", "jm039", "jm046"]
    assert decision["worst_candidate_minus_control_metric"] < 0.0


def test_subject_support_markdown_names_support_and_regressions() -> None:
    markdown = format_subject_support_markdown(evaluate_subject_support(_subject_rows()))

    assert "# FullMHT Identity-History Subject Support Decision" in markdown
    assert "stable_subject_support" in markdown
    assert "Complete-gain subjects" in markdown
    assert "worst control delta" in markdown


def test_subject_support_loads_labeled_csv_inputs(tmp_path) -> None:
    path = tmp_path / "candidate.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("subject", "pairwise_f1", "complete_track_f1"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "jm038",
                "pairwise_f1": "0.9",
                "complete_track_f1": "0.8",
            }
        )

    label, parsed_path = parse_labeled_input(f"FullMHTIdentityHistory={path}")
    rows = load_labeled_subject_rows([(label, parsed_path)])

    assert rows == [
        {
            "approach": "FullMHTIdentityHistory",
            "subject": "jm038",
            "pairwise_f1": "0.9",
            "complete_track_f1": "0.8",
        }
    ]
