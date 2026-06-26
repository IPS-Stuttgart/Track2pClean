from __future__ import annotations

import pytest

from bayescatrack.experiments.full_mht_growth_history_prediction_promotion_gate import (
    GrowthHistoryPredictionPromotionConfig,
    evaluate_growth_history_prediction_exposure_gate,
    evaluate_growth_history_prediction_promotion,
    format_promotion_markdown,
)


def _metric_row(approach: str, pairwise: float, complete: float) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise),
        "complete_track_f1_micro": str(complete),
        "pairwise_f1_macro": str(pairwise),
        "complete_track_f1_macro": str(complete),
    }


def _comparison_rows(*, stable_gain: bool = True) -> list[dict[str, str]]:
    complete_025 = 0.931 if stable_gain else 0.930
    complete_050 = 0.932 if stable_gain else 0.932
    return [
        _metric_row("Track2p", 0.962, 0.920),
        _metric_row("FullMHTPrior2", 0.965, 0.930),
        _metric_row("FullMHTGrowthHistoryPrediction025", 0.965, complete_025),
        _metric_row("FullMHTGrowthHistoryPrediction050", 0.965, complete_050),
        _metric_row("FullMHTGrowthHistoryPrediction100", 0.965, 0.930),
    ]


def _exposure_all_row(**overrides: float | int) -> dict[str, str]:
    values: dict[str, float | int | str] = {
        "subject": "ALL",
        "max_selected_non_prior_edges_per_subject": 2,
        "history_selected_non_prior_edges": 5,
        "history_growth_prediction_evaluated_edges": 40,
        "history_growth_prediction_penalized_edges": 4,
        "max_growth_prediction_penalized_edges_per_subject": 2,
        "history_growth_prediction_weighted_penalty": 2.5,
        "max_growth_prediction_weighted_penalty_per_subject": 1.25,
    }
    values.update(overrides)
    return {key: str(value) for key, value in values.items()}


def test_exposure_gate_accepts_bounded_growth_prediction_exposure() -> None:
    decision = evaluate_growth_history_prediction_exposure_gate([_exposure_all_row()])

    assert decision["status"] == "complete"
    assert decision["exposure_result"] == "bounded_exposure"
    assert decision["failed_limits"] == []
    assert decision["history_growth_prediction_evaluated_edges"] == 40


def test_exposure_gate_requires_growth_prediction_to_be_evaluated() -> None:
    decision = evaluate_growth_history_prediction_exposure_gate(
        [_exposure_all_row(history_growth_prediction_evaluated_edges=0)]
    )

    assert decision["status"] == "incomplete"
    assert decision["exposure_result"] == "growth_prediction_not_evaluated"


def test_exposure_gate_can_accept_zero_evaluated_edges_when_disabled() -> None:
    decision = evaluate_growth_history_prediction_exposure_gate(
        [_exposure_all_row(history_growth_prediction_evaluated_edges=0)],
        config=GrowthHistoryPredictionPromotionConfig(
            require_growth_prediction_evaluated=False
        ),
    )

    assert decision["status"] == "complete"
    assert decision["exposure_result"] == "bounded_exposure"


def test_exposure_gate_rejects_broad_growth_prediction_penalties() -> None:
    decision = evaluate_growth_history_prediction_exposure_gate(
        [
            _exposure_all_row(
                history_growth_prediction_penalized_edges=25,
                max_growth_prediction_penalized_edges_per_subject=5,
                history_growth_prediction_weighted_penalty=9.5,
                max_growth_prediction_weighted_penalty_per_subject=4.5,
            )
        ]
    )

    assert decision["exposure_result"] == "broad_exposure"
    assert "history_growth_prediction_penalized_edges" in decision["failed_limits"]
    assert "max_growth_prediction_penalized_edges_per_subject" in decision["failed_limits"]
    assert "history_growth_prediction_weighted_penalty" in decision["failed_limits"]
    assert "max_growth_prediction_weighted_penalty_per_subject" in decision["failed_limits"]


def test_exposure_gate_rejects_broad_non_prior_edges() -> None:
    decision = evaluate_growth_history_prediction_exposure_gate(
        [
            _exposure_all_row(
                max_selected_non_prior_edges_per_subject=4,
                history_selected_non_prior_edges=12,
            )
        ]
    )

    assert decision["exposure_result"] == "broad_exposure"
    assert "max_selected_non_prior_edges_per_subject" in decision["failed_limits"]
    assert "history_selected_non_prior_edges" in decision["failed_limits"]


def test_exposure_gate_reports_missing_all_row() -> None:
    decision = evaluate_growth_history_prediction_exposure_gate([{"subject": "jm_fake"}])

    assert decision["status"] == "incomplete"
    assert decision["exposure_result"] == "missing_all_row"


def test_promotion_gate_requires_stable_gain_and_bounded_exposure() -> None:
    decision = evaluate_growth_history_prediction_promotion(
        _comparison_rows(stable_gain=True),
        [_exposure_all_row()],
    )

    assert decision["status"] == "promotable_after_review"
    assert decision["benchmark_result"] == "history_dynamics_stable_gain"
    assert decision["exposure_result"] == "bounded_exposure"


def test_promotion_gate_rejects_stable_gain_with_broad_exposure() -> None:
    decision = evaluate_growth_history_prediction_promotion(
        _comparison_rows(stable_gain=True),
        [_exposure_all_row(history_growth_prediction_penalized_edges=30)],
    )

    assert decision["status"] == "not_promotable_broad_exposure"
    assert decision["benchmark_result"] == "history_dynamics_stable_gain"
    assert decision["exposure_result"] == "broad_exposure"


def test_promotion_gate_rejects_single_weight_gain() -> None:
    decision = evaluate_growth_history_prediction_promotion(
        _comparison_rows(stable_gain=False),
        [_exposure_all_row()],
    )

    assert decision["status"] == "not_promotable_no_stable_gain"
    assert decision["benchmark_result"] == "history_dynamics_single_weight_gain"


def test_promotion_gate_markdown_is_compact() -> None:
    markdown = format_promotion_markdown(
        evaluate_growth_history_prediction_promotion(
            _comparison_rows(stable_gain=True),
            [_exposure_all_row()],
        )
    )

    assert "# FullMHT Growth-History Prediction Promotion Gate" in markdown
    assert "promotable_after_review" in markdown
    assert "history_growth_prediction_penalized_edges" in markdown
    assert "max_growth_prediction_weighted_penalty_per_subject" in markdown


def test_exposure_gate_rejects_nonnumeric_metrics() -> None:
    with pytest.raises(ValueError, match="not numeric"):
        evaluate_growth_history_prediction_exposure_gate(
            [_exposure_all_row(history_growth_prediction_weighted_penalty="wide")]
        )
