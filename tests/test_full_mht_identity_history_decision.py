from __future__ import annotations

from bayescatrack.experiments.full_mht_identity_history_decision import (
    evaluate_identity_history_decision,
    format_decision_markdown,
)


def _row(approach: str, pairwise: float, complete: float) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise),
        "complete_track_f1_micro": str(complete),
        "pairwise_f1_macro": str(pairwise),
        "complete_track_f1_macro": str(complete),
    }


def _rows(
    *,
    identity_pairwise: float = 0.966,
    identity_complete: float = 0.934,
    greedy_pairwise: float = 0.966,
    greedy_complete: float = 0.931,
    prior_survival_pairwise: float = 0.966,
    prior_survival_complete: float = 0.932,
    no_prior_pairwise: float = 0.965,
    no_prior_complete: float = 0.930,
    no_local_pairwise: float = 0.965,
    no_local_complete: float = 0.932,
) -> list[dict[str, str]]:
    return [
        _row("Track2p", 0.965, 0.924),
        _row("FullMHTPrior2", 0.965, 0.930),
        _row("FullMHTPriorSurvival", prior_survival_pairwise, prior_survival_complete),
        _row("FullMHTNoPriorContinuation100", no_prior_pairwise, no_prior_complete),
        _row("FullMHTIdentityHistoryNoLocalContext", no_local_pairwise, no_local_complete),
        _row("FullMHTIdentityHistory", identity_pairwise, identity_complete),
        _row("FullMHTGreedyIdentityHistory", greedy_pairwise, greedy_complete),
    ]


def test_identity_history_decision_reports_missing_rows() -> None:
    decision = evaluate_identity_history_decision([_row("Track2p", 1.0, 1.0)])

    assert decision["status"] == "incomplete"
    assert "FullMHTIdentityHistory" in decision["missing_approaches"]
    assert "FullMHTIdentityHistoryNoLocalContext" in decision["missing_approaches"]
    assert "FullMHTGreedyIdentityHistory" in decision["missing_approaches"]


def test_identity_history_decision_detects_complete_history_advantage() -> None:
    decision = evaluate_identity_history_decision(_rows())

    assert decision["status"] == "complete"
    assert decision["mht_candidate"] == "FullMHTIdentityHistory"
    assert decision["local_choice_baseline"] == "FullMHTGreedyIdentityHistory"
    assert decision["no_local_context_baseline"] == "FullMHTIdentityHistoryNoLocalContext"
    assert decision["mht_vs_local_result"] == "identity_complete_history_advantage"
    assert decision["history_search_result"] == "identity_complete_history_advantage"
    assert decision["mht_minus_local_pairwise_f1_micro"] == 0.0
    assert decision["mht_minus_local_complete_track_f1_micro"] > 0.0
    assert decision["mht_minus_local_pairwise_f1_macro"] == 0.0
    assert decision["mht_minus_local_complete_track_f1_macro"] > 0.0
    assert decision["prior_control_result"] == "identity_improves_prior"
    assert decision["track2p_control_result"] == "identity_improves_track2p"
    assert decision["no_local_context_control_result"] == "identity_improves_no_local_context"
    assert decision["layer_combination_result"] == "combined_layer_gain"
    assert "promote only after" in decision["recommendation"]


def test_identity_history_decision_rejects_greedy_tie() -> None:
    decision = evaluate_identity_history_decision(
        _rows(identity_complete=0.931, greedy_complete=0.931)
    )

    assert decision["mht_vs_local_result"] == "identity_ties_greedy"
    assert decision["history_search_result"] == "identity_ties_greedy"
    assert decision["recommendation"].startswith("keep exploratory")


def test_identity_history_decision_rejects_pairwise_only_gain() -> None:
    decision = evaluate_identity_history_decision(
        _rows(
            identity_pairwise=0.967,
            identity_complete=0.931,
            greedy_pairwise=0.966,
            greedy_complete=0.931,
        )
    )

    assert decision["mht_vs_local_result"] == "identity_pairwise_only_advantage"
    assert decision["history_search_result"] == "identity_pairwise_only_advantage"
    assert "not complete-track advantage" in decision["recommendation"]


def test_identity_history_decision_rejects_macro_regression_against_greedy() -> None:
    rows = _rows(identity_complete=0.934, greedy_complete=0.931)
    for row in rows:
        if row["approach"] == "FullMHTIdentityHistory":
            row["complete_track_f1_macro"] = "0.900"
        elif row["approach"] == "FullMHTGreedyIdentityHistory":
            row["complete_track_f1_macro"] = "0.931"

    decision = evaluate_identity_history_decision(rows)

    assert decision["mht_minus_local_complete_track_f1_micro"] > 0.0
    assert decision["mht_minus_local_complete_track_f1_macro"] < 0.0
    assert decision["mht_vs_local_result"] == "identity_regression_vs_greedy"
    assert decision["recommendation"] == (
        "do not promote; identity-history beam regresses against matching greedy ablation"
    )


def test_identity_history_decision_rejects_required_control_regression() -> None:
    decision = evaluate_identity_history_decision(
        _rows(
            identity_pairwise=0.964,
            identity_complete=0.934,
            greedy_pairwise=0.963,
            greedy_complete=0.931,
            prior_survival_pairwise=0.964,
            prior_survival_complete=0.932,
        )
    )

    assert decision["history_search_result"] == "identity_complete_history_advantage"
    assert decision["prior_control_result"] == "identity_below_prior"
    assert decision["recommendation"] == (
        "keep exploratory; identity-history row loses to a required control"
    )


def test_identity_history_decision_rejects_macro_regression_against_control() -> None:
    rows = _rows(
        identity_pairwise=0.966,
        identity_complete=0.934,
        greedy_pairwise=0.963,
        prior_survival_pairwise=0.963,
        no_prior_pairwise=0.963,
        no_local_pairwise=0.963,
    )
    for row in rows:
        if row["approach"] == "FullMHTIdentityHistory":
            row["pairwise_f1_macro"] = "0.964"
        elif row["approach"] == "Track2p":
            row["pairwise_f1_macro"] = "0.965"
        elif row["approach"] == "FullMHTPrior2":
            row["pairwise_f1_macro"] = "0.963"

    decision = evaluate_identity_history_decision(rows)

    assert decision["mht_vs_local_result"] == "identity_complete_history_advantage"
    assert decision["identity_minus_track2p_pairwise_f1_micro"] > 0.0
    assert decision["identity_minus_track2p_pairwise_f1_macro"] < 0.0
    assert decision["track2p_control_result"] == "identity_below_track2p"
    assert decision["recommendation"] == (
        "keep exploratory; identity-history row loses to a required control"
    )


def test_identity_history_decision_rejects_component_layer_regression() -> None:
    decision = evaluate_identity_history_decision(
        _rows(
            identity_pairwise=0.966,
            identity_complete=0.934,
            prior_survival_pairwise=0.966,
            prior_survival_complete=0.936,
        )
    )

    assert decision["layer_combination_result"] == "combined_layer_regression"
    assert decision["recommendation"] == (
        "keep exploratory; combined history model falls below a component layer"
    )


def test_identity_history_decision_rejects_no_local_context_regression() -> None:
    decision = evaluate_identity_history_decision(
        _rows(
            identity_pairwise=0.966,
            identity_complete=0.934,
            no_local_pairwise=0.966,
            no_local_complete=0.936,
        )
    )

    assert decision["no_local_context_control_result"] == "identity_below_no_local_context"
    assert decision["recommendation"] == (
        "keep exploratory; calibrated local-context layer hurts identity-history row"
    )


def test_identity_history_decision_markdown_reports_all_four_metrics() -> None:
    markdown = format_decision_markdown(evaluate_identity_history_decision(_rows()))

    assert "# FullMHT Identity-History Decision" in markdown
    assert "MHT-vs-local result" in markdown
    assert "No-local-context result" in markdown
    assert "identity_complete_history_advantage" in markdown
    assert "MHT minus local greedy" in markdown
    assert "identity minus no-local context" in markdown
    assert "pairwise F1 macro delta" in markdown
    assert "complete-track F1 macro delta" in markdown
