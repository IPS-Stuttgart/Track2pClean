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
) -> list[dict[str, str]]:
    return [
        _row("Track2p", 0.965, 0.924),
        _row("FullMHTPrior2", 0.965, 0.930),
        _row("FullMHTPriorSurvival", prior_survival_pairwise, prior_survival_complete),
        _row("FullMHTNoPriorContinuation100", no_prior_pairwise, no_prior_complete),
        _row("FullMHTIdentityHistory", identity_pairwise, identity_complete),
        _row("FullMHTGreedyIdentityHistory", greedy_pairwise, greedy_complete),
    ]


def test_identity_history_decision_reports_missing_rows() -> None:
    decision = evaluate_identity_history_decision([_row("Track2p", 1.0, 1.0)])

    assert decision["status"] == "incomplete"
    assert "FullMHTIdentityHistory" in decision["missing_approaches"]
    assert "FullMHTGreedyIdentityHistory" in decision["missing_approaches"]


def test_identity_history_decision_detects_complete_history_advantage() -> None:
    decision = evaluate_identity_history_decision(_rows())

    assert decision["status"] == "complete"
    assert decision["mht_candidate"] == "FullMHTIdentityHistory"
    assert decision["local_choice_baseline"] == "FullMHTGreedyIdentityHistory"
    assert decision["mht_vs_local_result"] == "identity_complete_history_advantage"
    assert decision["history_search_result"] == "identity_complete_history_advantage"
    assert decision["mht_minus_local_pairwise_f1_micro"] == 0.0
    assert decision["mht_minus_local_complete_track_f1_micro"] > 0.0
    assert decision["prior_control_result"] == "identity_improves_prior"
    assert decision["track2p_control_result"] == "identity_improves_track2p"
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


def test_identity_history_decision_markdown_is_compact() -> None:
    markdown = format_decision_markdown(evaluate_identity_history_decision(_rows()))

    assert "# FullMHT Identity-History Decision" in markdown
    assert "MHT-vs-local result" in markdown
    assert "identity_complete_history_advantage" in markdown
    assert "MHT minus local greedy" in markdown
