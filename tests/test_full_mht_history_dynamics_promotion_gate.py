from __future__ import annotations

from bayescatrack.experiments.full_mht_history_dynamics_promotion_gate import (
    evaluate_exposure_gate,
    evaluate_history_dynamics_promotion,
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
        _metric_row("FullMHTHistoryDynamics025", 0.965, complete_025),
        _metric_row("FullMHTHistoryDynamics050", 0.965, complete_050),
        _metric_row("FullMHTHistoryDynamics100", 0.965, 0.930),
    ]


def _exposure_all_row(**overrides: int) -> dict[str, str]:
    values = {
        "subject": "ALL",
        "max_selected_non_prior_edges_per_subject": 2,
        "history_selected_non_prior_edges": 5,
        "history_switched_prior_successors": 0,
        "history_no_prior_successor_continuations": 6,
        "history_gap_reactivated_tracks": 1,
    }
    values.update({key: int(value) for key, value in overrides.items()})
    return {key: str(value) for key, value in values.items()}


def test_exposure_gate_accepts_bounded_exposure() -> None:
    decision = evaluate_exposure_gate([_exposure_all_row()])

    assert decision["status"] == "complete"
    assert decision["exposure_result"] == "bounded_exposure"
    assert decision["failed_limits"] == []


def test_exposure_gate_rejects_broad_non_prior_edits() -> None:
    decision = evaluate_exposure_gate(
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
    decision = evaluate_exposure_gate([{"subject": "jm_fake"}])

    assert decision["status"] == "incomplete"
    assert decision["exposure_result"] == "missing_all_row"


def test_promotion_gate_requires_stable_gain_and_bounded_exposure() -> None:
    decision = evaluate_history_dynamics_promotion(
        _comparison_rows(stable_gain=True),
        [_exposure_all_row()],
    )

    assert decision["status"] == "promotable_after_review"
    assert decision["benchmark_result"] == "history_dynamics_stable_gain"
    assert decision["exposure_result"] == "bounded_exposure"


def test_promotion_gate_rejects_stable_gain_with_broad_exposure() -> None:
    decision = evaluate_history_dynamics_promotion(
        _comparison_rows(stable_gain=True),
        [_exposure_all_row(history_selected_non_prior_edges=20)],
    )

    assert decision["status"] == "not_promotable_broad_exposure"
    assert decision["benchmark_result"] == "history_dynamics_stable_gain"
    assert decision["exposure_result"] == "broad_exposure"


def test_promotion_gate_rejects_single_weight_gain() -> None:
    decision = evaluate_history_dynamics_promotion(
        _comparison_rows(stable_gain=False),
        [_exposure_all_row()],
    )

    assert decision["status"] == "not_promotable_no_stable_gain"
    assert decision["benchmark_result"] == "history_dynamics_single_weight_gain"


def test_promotion_gate_markdown_is_compact() -> None:
    markdown = format_promotion_markdown(
        evaluate_history_dynamics_promotion(
            _comparison_rows(stable_gain=True),
            [_exposure_all_row()],
        )
    )

    assert "# FullMHT History Dynamics Promotion Gate" in markdown
    assert "promotable_after_review" in markdown
    assert "max_selected_non_prior_edges_per_subject" in markdown
