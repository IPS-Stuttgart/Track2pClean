from __future__ import annotations

from bayescatrack.experiments.full_mht_identity_history_scan_pruning_promotion_gate import (
    ScanPruningPromotionConfig,
    evaluate_scan_pruning_exposure_gate,
    evaluate_scan_pruning_promotion,
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
    complete_050 = 0.932
    return [
        _metric_row("FullMHTIdentityHistory", 0.965, 0.930),
        _metric_row("IdentityHistoryScanPruning025", 0.965, complete_025),
        _metric_row("GreedyIdentityHistoryScanPruning025", 0.965, 0.930),
        _metric_row("IdentityHistoryScanPruning050", 0.965, complete_050),
        _metric_row("GreedyIdentityHistoryScanPruning050", 0.965, 0.930),
        _metric_row("IdentityHistoryScanPruning100", 0.965, 0.930),
        _metric_row("GreedyIdentityHistoryScanPruning100", 0.965, 0.930),
    ]


def _exposure_all_row(**overrides: float) -> dict[str, str]:
    values: dict[str, float | str] = {
        "subject": "ALL",
        "max_selected_non_prior_edges_per_subject": 2,
        "history_selected_non_prior_edges": 5,
        "max_scan_motion_history_weighted_risk_per_subject": 4.0,
        "history_scan_motion_history_weighted_risk": 12.0,
    }
    values.update(overrides)
    return {key: str(value) for key, value in values.items()}


def test_scan_pruning_exposure_gate_accepts_bounded_exposure() -> None:
    decision = evaluate_scan_pruning_exposure_gate([_exposure_all_row()])

    assert decision["status"] == "complete"
    assert decision["exposure_result"] == "bounded_exposure"
    assert decision["failed_limits"] == []


def test_scan_pruning_exposure_gate_requires_scan_columns() -> None:
    decision = evaluate_scan_pruning_exposure_gate(
        [
            {
                "subject": "ALL",
                "max_selected_non_prior_edges_per_subject": "1",
                "history_selected_non_prior_edges": "2",
            }
        ]
    )

    assert decision["status"] == "incomplete"
    assert decision["exposure_result"] == "missing_scan_pruning_columns"
    assert "history_scan_motion_history_weighted_risk" in decision["missing_columns"]


def test_scan_pruning_exposure_gate_rejects_broad_weighted_risk() -> None:
    decision = evaluate_scan_pruning_exposure_gate(
        [
            _exposure_all_row(
                max_scan_motion_history_weighted_risk_per_subject=11.0,
                history_scan_motion_history_weighted_risk=30.0,
            )
        ]
    )

    assert decision["exposure_result"] == "broad_exposure"
    assert "max_scan_motion_history_weighted_risk_per_subject" in decision["failed_limits"]
    assert "history_scan_motion_history_weighted_risk" in decision["failed_limits"]


def test_scan_pruning_exposure_gate_uses_configurable_limits() -> None:
    decision = evaluate_scan_pruning_exposure_gate(
        [_exposure_all_row(history_scan_motion_history_weighted_risk=30.0)],
        config=ScanPruningPromotionConfig(max_total_scan_weighted_risk=35.0),
    )

    assert decision["exposure_result"] == "bounded_exposure"
    assert decision["limit_history_scan_motion_history_weighted_risk"] == 35.0


def test_scan_pruning_promotion_requires_stable_gain_and_bounded_exposure() -> None:
    decision = evaluate_scan_pruning_promotion(
        _comparison_rows(stable_gain=True),
        [_exposure_all_row()],
    )

    assert decision["status"] == "promotable_after_review"
    assert decision["benchmark_result"] == "scan_pruning_stable_complete_history_gain"
    assert decision["exposure_result"] == "bounded_exposure"


def test_scan_pruning_promotion_rejects_stable_gain_with_broad_exposure() -> None:
    decision = evaluate_scan_pruning_promotion(
        _comparison_rows(stable_gain=True),
        [_exposure_all_row(history_selected_non_prior_edges=20)],
    )

    assert decision["status"] == "not_promotable_broad_exposure"
    assert decision["benchmark_result"] == "scan_pruning_stable_complete_history_gain"
    assert decision["exposure_result"] == "broad_exposure"


def test_scan_pruning_promotion_rejects_single_weight_gain() -> None:
    decision = evaluate_scan_pruning_promotion(
        _comparison_rows(stable_gain=False),
        [_exposure_all_row()],
    )

    assert decision["status"] == "not_promotable_no_stable_gain"
    assert decision["benchmark_result"] == "scan_pruning_single_weight_gain"


def test_scan_pruning_promotion_markdown_is_compact() -> None:
    markdown = format_promotion_markdown(
        evaluate_scan_pruning_promotion(
            _comparison_rows(stable_gain=True),
            [_exposure_all_row()],
        )
    )

    assert "# FullMHT Identity-History Scan-Pruning Promotion Gate" in markdown
    assert "promotable_after_review" in markdown
    assert "max_scan_motion_history_weighted_risk_per_subject" in markdown
    assert "history_scan_motion_history_weighted_risk" in markdown
