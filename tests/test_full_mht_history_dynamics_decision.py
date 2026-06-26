from __future__ import annotations

from bayescatrack.experiments.full_mht_history_dynamics_decision import (
    evaluate_history_dynamics_decision,
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


def _probe_rows(
    *,
    c025: tuple[float, float] = (0.965, 0.931),
    c050: tuple[float, float] = (0.965, 0.932),
    c100: tuple[float, float] = (0.965, 0.930),
) -> list[dict[str, str]]:
    return [
        _row("Track2p", pairwise_micro=0.962, complete_micro=0.920),
        _row("FullMHTPrior2", pairwise_micro=0.965, complete_micro=0.930),
        _row(
            "FullMHTHistoryDynamics025",
            pairwise_micro=c025[0],
            complete_micro=c025[1],
        ),
        _row(
            "FullMHTHistoryDynamics050",
            pairwise_micro=c050[0],
            complete_micro=c050[1],
        ),
        _row(
            "FullMHTHistoryDynamics100",
            pairwise_micro=c100[0],
            complete_micro=c100[1],
        ),
    ]


def test_history_dynamics_decision_reports_missing_rows() -> None:
    decision = evaluate_history_dynamics_decision(
        [_row("FullMHTPrior2", pairwise_micro=1.0, complete_micro=1.0)]
    )

    assert decision["status"] == "incomplete"
    assert "FullMHTHistoryDynamics025" in decision["missing_approaches"]


def test_history_dynamics_decision_detects_stable_gain() -> None:
    decision = evaluate_history_dynamics_decision(_probe_rows())

    assert decision["status"] == "complete"
    assert decision["history_dynamics_result"] == "history_dynamics_stable_gain"
    assert decision["viable_candidate_count"] == 2
    assert decision["best_candidate"] == "FullMHTHistoryDynamics050"
    assert decision["best_candidate_complete_track_f1_micro_delta"] > 0.0
    assert "promote only after" in decision["recommendation"]


def test_history_dynamics_decision_flags_single_weight_gain() -> None:
    decision = evaluate_history_dynamics_decision(
        _probe_rows(c025=(0.965, 0.930), c050=(0.965, 0.932), c100=(0.965, 0.930))
    )

    assert decision["history_dynamics_result"] == "history_dynamics_single_weight_gain"
    assert decision["viable_candidate_count"] == 1
    assert decision["recommendation"].startswith("treat as exploratory")


def test_history_dynamics_decision_rejects_pairwise_regression() -> None:
    decision = evaluate_history_dynamics_decision(
        _probe_rows(c025=(0.964, 0.933), c050=(0.965, 0.930), c100=(0.965, 0.930))
    )

    assert decision["history_dynamics_result"] == "history_dynamics_pairwise_regression"
    assert decision["pairwise_regression_count"] == 1
    assert decision["candidate_decisions"][0]["decision"] == "pairwise_regression"


def test_history_dynamics_decision_rejects_regression_even_with_two_gains() -> None:
    decision = evaluate_history_dynamics_decision(
        _probe_rows(c025=(0.965, 0.931), c050=(0.965, 0.932), c100=(0.964, 0.933))
    )

    assert decision["history_dynamics_result"] == "history_dynamics_pairwise_regression"
    assert decision["viable_candidate_count"] == 2
    assert decision["pairwise_regression_count"] == 1


def test_history_dynamics_decision_rejects_macro_pairwise_regression() -> None:
    rows = _probe_rows(c025=(0.965, 0.933), c050=(0.965, 0.931), c100=(0.965, 0.931))
    rows[2]["pairwise_f1_macro"] = "0.900"

    decision = evaluate_history_dynamics_decision(rows)

    assert decision["candidate_decisions"][0]["delta_pairwise_f1_micro"] == 0.0
    assert decision["candidate_decisions"][0]["delta_pairwise_f1_macro"] < 0.0
    assert decision["candidate_decisions"][0]["decision"] == "pairwise_regression"
    assert decision["history_dynamics_result"] == "history_dynamics_pairwise_regression"


def test_history_dynamics_decision_rejects_complete_regression_even_with_two_gains() -> None:
    decision = evaluate_history_dynamics_decision(
        _probe_rows(c025=(0.965, 0.931), c050=(0.965, 0.932), c100=(0.965, 0.929))
    )

    assert decision["history_dynamics_result"] == "history_dynamics_complete_regression"
    assert decision["viable_candidate_count"] == 2
    assert decision["complete_regression_count"] == 1


def test_history_dynamics_decision_rejects_macro_complete_regression() -> None:
    rows = _probe_rows(c025=(0.965, 0.933), c050=(0.965, 0.931), c100=(0.965, 0.931))
    rows[2]["complete_track_f1_macro"] = "0.900"

    decision = evaluate_history_dynamics_decision(rows)

    assert decision["candidate_decisions"][0]["delta_complete_track_f1_micro"] > 0.0
    assert decision["candidate_decisions"][0]["delta_complete_track_f1_macro"] < 0.0
    assert decision["candidate_decisions"][0]["decision"] == "complete_regression"
    assert decision["history_dynamics_result"] == "history_dynamics_complete_regression"


def test_history_dynamics_decision_markdown_reports_all_four_metrics() -> None:
    markdown = format_decision_markdown(
        evaluate_history_dynamics_decision(_probe_rows())
    )

    assert "# FullMHT History Dynamics Decision" in markdown
    assert "FullMHTHistoryDynamics050" in markdown
    assert "history_dynamics_stable_gain" in markdown
    assert "pairwise F1 macro delta" in markdown
    assert "complete-track F1 macro delta" in markdown
