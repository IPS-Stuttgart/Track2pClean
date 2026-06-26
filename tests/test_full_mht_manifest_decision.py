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
    survival_pairwise: float = 0.966,
    survival_complete: float = 0.934,
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
            "FullMHTPriorSurvival",
            pairwise_micro=survival_pairwise,
            complete_micro=survival_complete,
        ),
    ]


def test_full_mht_manifest_decision_reports_missing_rows() -> None:
    decision = evaluate_full_mht_manifest_decision(
        [_row("Track2p", pairwise_micro=1.0, complete_micro=1.0)]
    )

    assert decision["status"] == "incomplete"
    assert "FullMHTGreedyPrior2" in decision["missing_approaches"]


def test_full_mht_manifest_decision_detects_history_advantage() -> None:
    decision = evaluate_full_mht_manifest_decision(_canonical_rows())

    assert decision["status"] == "complete"
    assert decision["history_search_result"] == "beam_history_advantage"
    assert decision["prior_survival_result"] == "survival_improves_fixed_veto"
    assert decision["beam_minus_greedy_complete_track_f1_micro"] > 0.0
    assert decision["survival_minus_veto_complete_track_f1_micro"] > 0.0
    assert "promote candidate only after" in decision["recommendation"]


def test_full_mht_manifest_decision_requires_real_history_advantage() -> None:
    decision = evaluate_full_mht_manifest_decision(
        _canonical_rows(beam_pairwise=0.965, beam_complete=0.924)
    )

    assert decision["history_search_result"] == "beam_ties_greedy"
    assert decision["recommendation"].startswith("keep FullMHT exploratory")


def test_full_mht_manifest_decision_marks_survival_below_veto() -> None:
    decision = evaluate_full_mht_manifest_decision(
        _canonical_rows(survival_pairwise=0.9655, survival_complete=0.930)
    )

    assert decision["history_search_result"] == "beam_history_advantage"
    assert decision["prior_survival_result"] == "survival_above_track2p_but_below_fixed_veto"
    assert decision["recommendation"].startswith("keep prior-survival exploratory")


def test_full_mht_manifest_decision_markdown_is_compact() -> None:
    markdown = format_decision_markdown(
        evaluate_full_mht_manifest_decision(_canonical_rows())
    )

    assert "# FullMHT Manifest Decision" in markdown
    assert "beam minus greedy" in markdown
    assert "survival minus veto" in markdown
