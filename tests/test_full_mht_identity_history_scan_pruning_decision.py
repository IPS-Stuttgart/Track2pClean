from __future__ import annotations

from bayescatrack.experiments.full_mht_identity_history_scan_pruning_decision import (
    evaluate_identity_history_scan_pruning_decision,
    format_scan_pruning_decision_markdown,
)


def _row(
    approach: str,
    *,
    pairwise: float,
    complete: float,
    pairwise_macro: float | None = None,
    complete_macro: float | None = None,
) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise),
        "complete_track_f1_micro": str(complete),
        "pairwise_f1_macro": str(pairwise if pairwise_macro is None else pairwise_macro),
        "complete_track_f1_macro": str(complete if complete_macro is None else complete_macro),
    }


def _rows(
    *,
    c025: tuple[float, float] = (0.966, 0.934),
    g025: tuple[float, float] = (0.966, 0.931),
    c050: tuple[float, float] = (0.966, 0.935),
    g050: tuple[float, float] = (0.966, 0.932),
    c100: tuple[float, float] = (0.966, 0.933),
    g100: tuple[float, float] = (0.966, 0.933),
) -> list[dict[str, str]]:
    return [
        _row("FullMHTIdentityHistory", pairwise=0.966, complete=0.933),
        _row("IdentityHistoryScanPruning025", pairwise=c025[0], complete=c025[1]),
        _row("GreedyIdentityHistoryScanPruning025", pairwise=g025[0], complete=g025[1]),
        _row("IdentityHistoryScanPruning050", pairwise=c050[0], complete=c050[1]),
        _row("GreedyIdentityHistoryScanPruning050", pairwise=g050[0], complete=g050[1]),
        _row("IdentityHistoryScanPruning100", pairwise=c100[0], complete=c100[1]),
        _row("GreedyIdentityHistoryScanPruning100", pairwise=g100[0], complete=g100[1]),
    ]


def test_scan_pruning_decision_reports_missing_rows() -> None:
    decision = evaluate_identity_history_scan_pruning_decision(
        [_row("FullMHTIdentityHistory", pairwise=1.0, complete=1.0)]
    )

    assert decision["status"] == "incomplete"
    assert decision["scan_pruning_result"] == "missing_rows"
    assert "IdentityHistoryScanPruning025" in decision["missing_approaches"]


def test_scan_pruning_decision_detects_stable_complete_history_gain() -> None:
    decision = evaluate_identity_history_scan_pruning_decision(_rows())

    assert decision["status"] == "complete"
    assert decision["scan_pruning_result"] == "scan_pruning_stable_complete_history_gain"
    assert decision["viable_candidate_count"] == 2
    assert decision["best_candidate"] == "IdentityHistoryScanPruning050"
    assert "promote scan-history pruning only after" in decision["recommendation"]


def test_scan_pruning_decision_flags_single_weight_gain() -> None:
    decision = evaluate_identity_history_scan_pruning_decision(
        _rows(c025=(0.966, 0.933), g025=(0.966, 0.933), c050=(0.966, 0.935), g050=(0.966, 0.932), c100=(0.966, 0.933), g100=(0.966, 0.933))
    )

    assert decision["scan_pruning_result"] == "scan_pruning_single_weight_gain"
    assert decision["viable_candidate_count"] == 1


def test_scan_pruning_decision_rejects_pairwise_only_gain() -> None:
    decision = evaluate_identity_history_scan_pruning_decision(
        _rows(c025=(0.967, 0.933), g025=(0.966, 0.933), c050=(0.966, 0.933), g050=(0.966, 0.933), c100=(0.966, 0.933), g100=(0.966, 0.933))
    )

    assert decision["scan_pruning_result"] == "scan_pruning_pairwise_only_gain"
    assert decision["candidate_decisions"][0]["decision"] == "pairwise_only_gain"


def test_scan_pruning_decision_rejects_beam_regression_vs_greedy() -> None:
    decision = evaluate_identity_history_scan_pruning_decision(
        _rows(c025=(0.965, 0.934), g025=(0.966, 0.933))
    )

    assert decision["scan_pruning_result"] == "scan_pruning_beam_regression_vs_greedy"
    assert decision["candidate_decisions"][0]["decision"] == "beam_regression_vs_greedy"


def test_scan_pruning_decision_rejects_regression_vs_identity_history_baseline() -> None:
    decision = evaluate_identity_history_scan_pruning_decision(
        _rows(c025=(0.966, 0.932), g025=(0.966, 0.930), c050=(0.966, 0.935), g050=(0.966, 0.932))
    )

    assert decision["candidate_decisions"][0]["decision"] == "scan_pruning_regression_vs_identity_history"
    assert decision["scan_pruning_result"] == "scan_pruning_regression_vs_identity_history"


def test_scan_pruning_decision_rejects_macro_complete_regression() -> None:
    rows = _rows()
    for row in rows:
        if row["approach"] == "IdentityHistoryScanPruning025":
            row["complete_track_f1_macro"] = "0.920"
        if row["approach"] == "GreedyIdentityHistoryScanPruning025":
            row["complete_track_f1_macro"] = "0.931"

    decision = evaluate_identity_history_scan_pruning_decision(rows)

    assert decision["candidate_decisions"][0]["beam_minus_greedy_complete_track_f1_micro"] > 0.0
    assert decision["candidate_decisions"][0]["beam_minus_greedy_complete_track_f1_macro"] < 0.0
    assert decision["candidate_decisions"][0]["decision"] == "beam_regression_vs_greedy"
    assert decision["scan_pruning_result"] == "scan_pruning_beam_regression_vs_greedy"


def test_scan_pruning_decision_markdown_reports_candidates() -> None:
    markdown = format_scan_pruning_decision_markdown(
        evaluate_identity_history_scan_pruning_decision(_rows())
    )

    assert "# FullMHT Identity-History Scan-Pruning Decision" in markdown
    assert "IdentityHistoryScanPruning050" in markdown
    assert "scan_pruning_stable_complete_history_gain" in markdown
    assert "vs baseline complete micro" in markdown
