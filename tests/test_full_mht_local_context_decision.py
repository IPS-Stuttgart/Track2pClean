from __future__ import annotations

from bayescatrack.experiments.full_mht_local_context_decision import (
    evaluate_local_context_decision,
    format_local_context_markdown,
)


def _metric_row(approach: str, pairwise: float, complete: float) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise),
        "complete_track_f1_micro": str(complete),
        "pairwise_f1_macro": str(pairwise),
        "complete_track_f1_macro": str(complete),
    }


def test_local_context_decision_uses_probe_row_names() -> None:
    decision = evaluate_local_context_decision(
        [
            _metric_row("Track2p", 0.962, 0.920),
            _metric_row("FullMHTLocalContext000", 0.965, 0.930),
            _metric_row("FullMHTLocalContext025", 0.965, 0.931),
            _metric_row("FullMHTLocalContext050", 0.965, 0.932),
            _metric_row("FullMHTLocalContext100", 0.965, 0.930),
        ]
    )

    assert decision["status"] == "complete"
    assert decision["local_context_result"] == "history_dynamics_stable_gain"
    assert decision["best_candidate"] == "FullMHTLocalContext050"
    assert "neighborhood-coherence" in decision["recommendation"]


def test_local_context_decision_reports_missing_rows() -> None:
    decision = evaluate_local_context_decision(
        [_metric_row("FullMHTLocalContext000", 0.965, 0.930)]
    )

    assert decision["status"] == "incomplete"
    assert "FullMHTLocalContext025" in decision["missing_approaches"]


def test_local_context_decision_flags_single_weight_gain() -> None:
    decision = evaluate_local_context_decision(
        [
            _metric_row("Track2p", 0.962, 0.920),
            _metric_row("FullMHTLocalContext000", 0.965, 0.930),
            _metric_row("FullMHTLocalContext025", 0.965, 0.930),
            _metric_row("FullMHTLocalContext050", 0.965, 0.932),
            _metric_row("FullMHTLocalContext100", 0.965, 0.930),
        ]
    )

    assert decision["local_context_result"] == "history_dynamics_single_weight_gain"
    assert decision["recommendation"].startswith("treat as exploratory")


def test_local_context_decision_rejects_pairwise_regression() -> None:
    decision = evaluate_local_context_decision(
        [
            _metric_row("Track2p", 0.962, 0.920),
            _metric_row("FullMHTLocalContext000", 0.965, 0.930),
            _metric_row("FullMHTLocalContext025", 0.964, 0.933),
            _metric_row("FullMHTLocalContext050", 0.965, 0.930),
            _metric_row("FullMHTLocalContext100", 0.965, 0.930),
        ]
    )

    assert decision["local_context_result"] == "history_dynamics_pairwise_regression"
    assert decision["pairwise_regression_count"] == 1


def test_local_context_decision_markdown_is_specific() -> None:
    markdown = format_local_context_markdown(
        evaluate_local_context_decision(
            [
                _metric_row("Track2p", 0.962, 0.920),
                _metric_row("FullMHTLocalContext000", 0.965, 0.930),
                _metric_row("FullMHTLocalContext025", 0.965, 0.931),
                _metric_row("FullMHTLocalContext050", 0.965, 0.932),
                _metric_row("FullMHTLocalContext100", 0.965, 0.930),
            ]
        )
    )

    assert "# FullMHT Local-Context Decision" in markdown
    assert "FullMHTLocalContext050" in markdown
    assert "history_dynamics_stable_gain" in markdown
