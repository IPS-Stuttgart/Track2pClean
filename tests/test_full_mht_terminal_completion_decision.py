from __future__ import annotations

from bayescatrack.experiments.full_mht_terminal_completion_decision import (
    TerminalCompletionDecisionConfig,
    build_arg_parser,
    evaluate_terminal_completion_decision,
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
            "FullMHTTerminalCompletion025",
            pairwise_micro=c025[0],
            complete_micro=c025[1],
        ),
        _row(
            "FullMHTTerminalCompletion050",
            pairwise_micro=c050[0],
            complete_micro=c050[1],
        ),
        _row(
            "FullMHTTerminalCompletion100",
            pairwise_micro=c100[0],
            complete_micro=c100[1],
        ),
    ]


def test_terminal_completion_decision_reports_missing_rows() -> None:
    decision = evaluate_terminal_completion_decision(
        [_row("FullMHTPrior2", pairwise_micro=1.0, complete_micro=1.0)]
    )

    assert decision["status"] == "incomplete"
    assert "FullMHTTerminalCompletion025" in decision["missing_approaches"]


def test_terminal_completion_decision_detects_stable_gain() -> None:
    decision = evaluate_terminal_completion_decision(_probe_rows())

    assert decision["status"] == "complete"
    assert decision["terminal_completion_result"] == "terminal_completion_stable_gain"
    assert decision["viable_candidate_count"] == 2
    assert decision["best_candidate"] == "FullMHTTerminalCompletion050"
    assert decision["best_candidate_complete_track_f1_micro_delta"] > 0.0
    assert "promote only after" in decision["recommendation"]


def test_terminal_completion_decision_flags_single_weight_gain() -> None:
    decision = evaluate_terminal_completion_decision(
        _probe_rows(c025=(0.965, 0.930), c050=(0.965, 0.932), c100=(0.965, 0.930))
    )

    assert decision["terminal_completion_result"] == "terminal_completion_single_weight_gain"
    assert decision["viable_candidate_count"] == 1
    assert decision["recommendation"].startswith("treat as exploratory")


def test_terminal_completion_decision_rejects_pairwise_regression() -> None:
    decision = evaluate_terminal_completion_decision(
        _probe_rows(c025=(0.964, 0.933), c050=(0.965, 0.930), c100=(0.965, 0.930))
    )

    assert decision["terminal_completion_result"] == "terminal_completion_pairwise_regression"
    assert decision["pairwise_regression_count"] == 1
    assert decision["candidate_decisions"][0]["decision"] == "pairwise_regression"


def test_terminal_completion_decision_rejects_regression_even_with_two_gains() -> None:
    decision = evaluate_terminal_completion_decision(
        _probe_rows(c025=(0.965, 0.931), c050=(0.965, 0.932), c100=(0.964, 0.933))
    )

    assert decision["terminal_completion_result"] == "terminal_completion_pairwise_regression"
    assert decision["viable_candidate_count"] == 2
    assert decision["pairwise_regression_count"] == 1


def test_terminal_completion_decision_rejects_complete_regression_even_with_two_gains() -> None:
    decision = evaluate_terminal_completion_decision(
        _probe_rows(c025=(0.965, 0.931), c050=(0.965, 0.932), c100=(0.965, 0.929))
    )

    assert decision["terminal_completion_result"] == "terminal_completion_complete_regression"
    assert decision["viable_candidate_count"] == 2
    assert decision["complete_regression_count"] == 1


def test_terminal_completion_decision_accepts_custom_identity_history_rows() -> None:
    rows = [
        _row("Track2p", pairwise_micro=0.962, complete_micro=0.920),
        _row("FullMHTIdentityHistory", pairwise_micro=0.965, complete_micro=0.932),
        _row("FullMHTIdentityHistoryCompletion025", pairwise_micro=0.965, complete_micro=0.933),
        _row("FullMHTIdentityHistoryCompletion050", pairwise_micro=0.965, complete_micro=0.934),
        _row("FullMHTIdentityHistoryCompletion100", pairwise_micro=0.965, complete_micro=0.932),
    ]

    decision = evaluate_terminal_completion_decision(
        rows,
        config=TerminalCompletionDecisionConfig(
            baseline="FullMHTIdentityHistory",
            candidates=(
                "FullMHTIdentityHistoryCompletion025",
                "FullMHTIdentityHistoryCompletion050",
                "FullMHTIdentityHistoryCompletion100",
            ),
        ),
    )

    assert decision["baseline"] == "FullMHTIdentityHistory"
    assert decision["terminal_completion_result"] == "terminal_completion_stable_gain"
    assert decision["best_candidate"] == "FullMHTIdentityHistoryCompletion050"


def test_terminal_completion_decision_parser_accepts_row_overrides() -> None:
    args = build_arg_parser().parse_args(
        [
            "comparison.csv",
            "--baseline",
            "FullMHTIdentityHistory",
            "--candidate",
            "FullMHTIdentityHistoryCompletion025",
            "--candidate",
            "FullMHTIdentityHistoryCompletion050",
            "--candidate",
            "FullMHTIdentityHistoryCompletion100",
        ]
    )

    assert args.baseline == "FullMHTIdentityHistory"
    assert args.candidates == [
        "FullMHTIdentityHistoryCompletion025",
        "FullMHTIdentityHistoryCompletion050",
        "FullMHTIdentityHistoryCompletion100",
    ]


def test_terminal_completion_decision_markdown_is_compact() -> None:
    markdown = format_decision_markdown(
        evaluate_terminal_completion_decision(_probe_rows())
    )

    assert "# FullMHT Terminal Completion Decision" in markdown
    assert "Baseline: `FullMHTPrior2`" in markdown
    assert "FullMHTTerminalCompletion050" in markdown
    assert "terminal_completion_stable_gain" in markdown
