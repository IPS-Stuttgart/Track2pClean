from __future__ import annotations

from bayescatrack.experiments.full_mht_manifest_decision import (
    evaluate_full_mht_manifest_decision,
    format_decision_markdown,
)


def _row(
    approach: str,
    *,
    pairwise_micro: float,
    complete_micro: float,
    pairwise_macro: float | None = None,
    complete_macro: float | None = None,
) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise_micro),
        "complete_track_f1_micro": str(complete_micro),
        "pairwise_f1_macro": str(pairwise_macro if pairwise_macro is not None else pairwise_micro),
        "complete_track_f1_macro": str(
            complete_macro if complete_macro is not None else complete_micro
        ),
    }


def _canonical_rows(
    *,
    greedy_pairwise: float = 0.965,
    greedy_complete: float = 0.924,
    beam_pairwise: float = 0.965,
    beam_complete: float = 0.930,
    veto_pairwise: float = 0.966,
    veto_complete: float = 0.932,
    greedy_veto_pairwise: float = 0.966,
    greedy_veto_complete: float = 0.929,
    survival_pairwise: float = 0.966,
    survival_complete: float = 0.934,
    greedy_survival_pairwise: float = 0.966,
    greedy_survival_complete: float = 0.931,
) -> list[dict[str, str]]:
    return [
        _row("Track2p", pairwise_micro=0.965, complete_micro=0.924),
        _row("FullMHTPrior2", pairwise_micro=beam_pairwise, complete_micro=beam_complete),
        _row(
            "FullMHTGreedyPrior2",
            pairwise_micro=greedy_pairwise,
            complete_micro=greedy_complete,
        ),
        _row(
            "FullMHTPriorVetoScaled", pairwise_micro=veto_pairwise, complete_micro=veto_complete
        ),
        _row(
            "FullMHTGreedyPriorVetoScaled",
            pairwise_micro=greedy_veto_pairwise,
            complete_micro=greedy_veto_complete,
        ),
        _row(
            "FullMHTPriorSurvival",
            pairwise_micro=survival_pairwise,
            complete_micro=survival_complete,
        ),
        _row(
            "FullMHTGreedyPriorSurvival",
            pairwise_micro=greedy_survival_pairwise,
            complete_micro=greedy_survival_complete,
        ),
    ]


def test_full_mht_manifest_decision_reports_missing_rows() -> None:
    decision = evaluate_full_mht_manifest_decision(
        [_row("Track2p", pairwise_micro=1.0, complete_micro=1.0)]
    )

    assert decision["status"] == "incomplete"
    assert "FullMHTGreedyPrior2" in decision["missing_approaches"]
    assert "FullMHTGreedyPriorVetoScaled" in decision["missing_approaches"]
    assert "FullMHTGreedyPriorSurvival" in decision["missing_approaches"]


def test_full_mht_manifest_decision_detects_survival_complete_history_advantage() -> None:
    decision = evaluate_full_mht_manifest_decision(_canonical_rows())

    assert decision["status"] == "complete"
    assert decision["base_history_search_result"] == "beam_complete_history_advantage"
    assert decision["prior_veto_history_search_result"] == "beam_complete_history_advantage"
    assert decision["prior_survival_history_search_result"] == "beam_complete_history_advantage"
    assert decision["history_search_result"] == "prior_survival_complete_history_advantage"
    assert decision["prior_survival_result"] == "survival_improves_fixed_veto"
    assert decision["survival_beam_minus_greedy_complete_track_f1_micro"] > 0.0
    assert "promote prior-survival candidate only after" in decision["recommendation"]


def test_full_mht_manifest_decision_can_fall_back_to_fixed_veto_history_row() -> None:
    decision = evaluate_full_mht_manifest_decision(
        _canonical_rows(
            survival_pairwise=0.9655,
            survival_complete=0.930,
            greedy_survival_pairwise=0.9655,
            greedy_survival_complete=0.930,
        )
    )

    assert decision["history_search_result"] == "fixed_veto_complete_history_advantage"
    assert decision["prior_survival_history_search_result"] == "beam_ties_greedy"
    assert decision["prior_survival_result"] == "survival_above_track2p_but_below_fixed_veto"
    assert "fixed prior-veto is the current FullMHT history-search row" in decision["recommendation"]


def test_full_mht_manifest_decision_requires_candidate_complete_history_advantage() -> None:
    decision = evaluate_full_mht_manifest_decision(
        _canonical_rows(
            veto_pairwise=0.967,
            veto_complete=0.929,
            greedy_veto_pairwise=0.966,
            greedy_veto_complete=0.929,
            survival_pairwise=0.967,
            survival_complete=0.929,
            greedy_survival_pairwise=0.966,
            greedy_survival_complete=0.929,
        )
    )

    assert decision["history_search_result"] == "candidate_pairwise_only_advantage"
    assert "not complete-history advantage" in decision["recommendation"]


def test_full_mht_manifest_decision_requires_real_candidate_history_advantage() -> None:
    decision = evaluate_full_mht_manifest_decision(
        _canonical_rows(
            veto_pairwise=0.966,
            veto_complete=0.929,
            greedy_veto_pairwise=0.966,
            greedy_veto_complete=0.929,
            survival_pairwise=0.966,
            survival_complete=0.931,
            greedy_survival_pairwise=0.966,
            greedy_survival_complete=0.931,
        )
    )

    assert decision["history_search_result"] == "candidate_ties_greedy"
    assert decision["recommendation"].startswith("keep FullMHT exploratory")


def test_full_mht_manifest_decision_rejects_candidate_beam_regression() -> None:
    decision = evaluate_full_mht_manifest_decision(
        _canonical_rows(
            veto_pairwise=0.965,
            veto_complete=0.928,
            greedy_veto_pairwise=0.966,
            greedy_veto_complete=0.929,
            survival_pairwise=0.965,
            survival_complete=0.928,
            greedy_survival_pairwise=0.966,
            greedy_survival_complete=0.929,
        )
    )

    assert decision["history_search_result"] == "candidate_regression_vs_greedy"
    assert decision["recommendation"].startswith("do not promote FullMHT")


def test_full_mht_manifest_decision_rejects_macro_pairwise_regression() -> None:
    rows = _canonical_rows(
        veto_pairwise=0.966,
        veto_complete=0.929,
        greedy_veto_pairwise=0.966,
        greedy_veto_complete=0.929,
        survival_pairwise=0.966,
        survival_complete=0.934,
        greedy_survival_pairwise=0.966,
        greedy_survival_complete=0.931,
    )
    for row in rows:
        if row["approach"] == "FullMHTPriorSurvival":
            row["pairwise_f1_macro"] = "0.900"
        if row["approach"] == "FullMHTGreedyPriorSurvival":
            row["pairwise_f1_macro"] = "0.965"

    decision = evaluate_full_mht_manifest_decision(rows)

    assert decision["prior_survival_history_search_result"] == "beam_regression_vs_greedy"
    assert decision["history_search_result"] == "candidate_regression_vs_greedy"
    assert decision["survival_beam_minus_greedy_pairwise_f1_micro"] == 0.0
    assert decision["survival_beam_minus_greedy_pairwise_f1_macro"] < 0.0


def test_full_mht_manifest_decision_rejects_macro_complete_regression() -> None:
    rows = _canonical_rows(
        veto_pairwise=0.966,
        veto_complete=0.929,
        greedy_veto_pairwise=0.966,
        greedy_veto_complete=0.929,
        survival_pairwise=0.966,
        survival_complete=0.934,
        greedy_survival_pairwise=0.966,
        greedy_survival_complete=0.931,
    )
    for row in rows:
        if row["approach"] == "FullMHTPriorSurvival":
            row["complete_track_f1_macro"] = "0.900"
        if row["approach"] == "FullMHTGreedyPriorSurvival":
            row["complete_track_f1_macro"] = "0.931"

    decision = evaluate_full_mht_manifest_decision(rows)

    assert decision["prior_survival_history_search_result"] == "beam_regression_vs_greedy"
    assert decision["history_search_result"] == "candidate_regression_vs_greedy"
    assert decision["survival_beam_minus_greedy_complete_track_f1_micro"] > 0.0
    assert decision["survival_beam_minus_greedy_complete_track_f1_macro"] < 0.0


def test_full_mht_manifest_decision_markdown_is_compact() -> None:
    markdown = format_decision_markdown(
        evaluate_full_mht_manifest_decision(_canonical_rows())
    )

    assert "# FullMHT Manifest Decision" in markdown
    assert "base beam minus greedy" in markdown
    assert "veto beam minus greedy" in markdown
    assert "survival beam minus greedy" in markdown
    assert "pairwise F1 macro delta" in markdown
    assert "complete-track F1 macro delta" in markdown
    assert "prior_survival_complete_history_advantage" in markdown
