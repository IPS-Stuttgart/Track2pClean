from __future__ import annotations

from bayescatrack.experiments.full_mht_growth_history_prediction_decision import (
    evaluate_growth_history_prediction_decision,
    format_growth_history_prediction_markdown,
)


def _metric_row(approach: str, pairwise: float, complete: float) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise),
        "complete_track_f1_micro": str(complete),
        "pairwise_f1_macro": str(pairwise),
        "complete_track_f1_macro": str(complete),
    }


def test_growth_history_prediction_decision_uses_growth_probe_row_names() -> None:
    decision = evaluate_growth_history_prediction_decision(
        [
            _metric_row("Track2p", 0.962, 0.920),
            _metric_row("FullMHTPrior2", 0.965, 0.930),
            _metric_row("FullMHTGrowthHistoryPrediction025", 0.965, 0.931),
            _metric_row("FullMHTGrowthHistoryPrediction050", 0.965, 0.932),
            _metric_row("FullMHTGrowthHistoryPrediction100", 0.965, 0.930),
        ]
    )

    assert decision["status"] == "complete"
    assert decision["growth_history_prediction_result"] == "history_dynamics_stable_gain"
    assert decision["best_candidate"] == "FullMHTGrowthHistoryPrediction050"
    assert "scan-assignment dynamics component" in decision["recommendation"]


def test_growth_history_prediction_decision_reports_missing_rows() -> None:
    decision = evaluate_growth_history_prediction_decision(
        [_metric_row("FullMHTPrior2", 0.965, 0.930)]
    )

    assert decision["status"] == "incomplete"
    assert "FullMHTGrowthHistoryPrediction025" in decision["missing_approaches"]


def test_growth_history_prediction_decision_flags_single_weight_gain() -> None:
    decision = evaluate_growth_history_prediction_decision(
        [
            _metric_row("Track2p", 0.962, 0.920),
            _metric_row("FullMHTPrior2", 0.965, 0.930),
            _metric_row("FullMHTGrowthHistoryPrediction025", 0.965, 0.930),
            _metric_row("FullMHTGrowthHistoryPrediction050", 0.965, 0.932),
            _metric_row("FullMHTGrowthHistoryPrediction100", 0.965, 0.930),
        ]
    )

    assert decision["growth_history_prediction_result"] == "history_dynamics_single_weight_gain"
    assert decision["recommendation"].startswith("treat as exploratory")


def test_growth_history_prediction_decision_rejects_pairwise_regression() -> None:
    decision = evaluate_growth_history_prediction_decision(
        [
            _metric_row("Track2p", 0.962, 0.920),
            _metric_row("FullMHTPrior2", 0.965, 0.930),
            _metric_row("FullMHTGrowthHistoryPrediction025", 0.964, 0.933),
            _metric_row("FullMHTGrowthHistoryPrediction050", 0.965, 0.930),
            _metric_row("FullMHTGrowthHistoryPrediction100", 0.965, 0.930),
        ]
    )

    assert decision["growth_history_prediction_result"] == "history_dynamics_pairwise_regression"
    assert decision["pairwise_regression_count"] == 1


def test_growth_history_prediction_decision_markdown_is_specific() -> None:
    markdown = format_growth_history_prediction_markdown(
        evaluate_growth_history_prediction_decision(
            [
                _metric_row("Track2p", 0.962, 0.920),
                _metric_row("FullMHTPrior2", 0.965, 0.930),
                _metric_row("FullMHTGrowthHistoryPrediction025", 0.965, 0.931),
                _metric_row("FullMHTGrowthHistoryPrediction050", 0.965, 0.932),
                _metric_row("FullMHTGrowthHistoryPrediction100", 0.965, 0.930),
            ]
        )
    )

    assert "# FullMHT Growth-History Prediction Decision" in markdown
    assert "FullMHTGrowthHistoryPrediction050" in markdown
    assert "history_dynamics_stable_gain" in markdown
