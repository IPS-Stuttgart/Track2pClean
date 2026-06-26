from __future__ import annotations

from bayescatrack.experiments.full_mht_no_prior_continuation_promotion_gate import (
    evaluate_no_prior_continuation_exposure,
    evaluate_no_prior_continuation_promotion,
    format_no_prior_continuation_promotion_markdown,
)


def _metric_row(approach: str, pairwise: float, complete: float) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise),
        "complete_track_f1_micro": str(complete),
        "pairwise_f1_macro": str(pairwise),
        "complete_track_f1_macro": str(complete),
    }


def _stable_gain_rows() -> list[dict[str, str]]:
    return [
        _metric_row("Track2p", 0.962, 0.920),
        _metric_row("FullMHTPrior2", 0.965, 0.930),
        _metric_row("FullMHTCalibratedNoDeath", 0.950, 0.850),
        _metric_row("FullMHTNoPriorContinuation050", 0.965, 0.931),
        _metric_row("FullMHTNoPriorContinuation100", 0.965, 0.932),
        _metric_row("FullMHTNoPriorContinuation150", 0.965, 0.930),
    ]


def _single_weight_gain_rows() -> list[dict[str, str]]:
    return [
        _metric_row("Track2p", 0.962, 0.920),
        _metric_row("FullMHTPrior2", 0.965, 0.930),
        _metric_row("FullMHTCalibratedNoDeath", 0.950, 0.850),
        _metric_row("FullMHTNoPriorContinuation050", 0.965, 0.930),
        _metric_row("FullMHTNoPriorContinuation100", 0.965, 0.932),
        _metric_row("FullMHTNoPriorContinuation150", 0.965, 0.930),
    ]


def _exposure_row(**overrides: object) -> dict[str, str]:
    row = {
        "subject": "ALL",
        "max_selected_non_prior_edges_per_subject": "2",
        "history_selected_non_prior_edges": "4",
        "history_switched_prior_successors": "0",
        "history_no_prior_successor_continuations": "4",
        "history_gap_reactivated_tracks": "0",
        "max_no_prior_continuation_scored_edges_per_subject": "3",
        "max_no_prior_continuation_positive_edges_per_subject": "2",
        "history_no_prior_continuation_scored_edges": "5",
        "history_no_prior_continuation_positive_edges": "3",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def test_no_prior_continuation_exposure_gate_accepts_bounded_exposure() -> None:
    exposure = evaluate_no_prior_continuation_exposure([_exposure_row()])

    assert exposure["status"] == "complete"
    assert exposure["exposure_result"] == "bounded_exposure"
    assert exposure["failed_limits"] == []


def test_no_prior_continuation_exposure_gate_rejects_broad_positive_firing() -> None:
    exposure = evaluate_no_prior_continuation_exposure(
        [
            _exposure_row(
                max_no_prior_continuation_positive_edges_per_subject=5,
                history_no_prior_continuation_positive_edges=12,
            )
        ]
    )

    assert exposure["exposure_result"] == "broad_exposure"
    assert "max_no_prior_continuation_positive_edges_per_subject" in exposure["failed_limits"]
    assert "history_no_prior_continuation_positive_edges" in exposure["failed_limits"]


def test_no_prior_continuation_promotion_requires_stable_gain_and_bounded_exposure() -> None:
    decision = evaluate_no_prior_continuation_promotion(
        _stable_gain_rows(),
        [_exposure_row()],
    )

    assert decision["status"] == "promotable_after_review"
    assert decision["benchmark_result"] == "no_prior_continuation_stable_gain"
    assert decision["exposure_result"] == "bounded_exposure"


def test_no_prior_continuation_promotion_rejects_broad_exposure() -> None:
    decision = evaluate_no_prior_continuation_promotion(
        _stable_gain_rows(),
        [_exposure_row(history_no_prior_successor_continuations=99)],
    )

    assert decision["status"] == "not_promotable_broad_exposure"
    assert decision["exposure_result"] == "broad_exposure"


def test_no_prior_continuation_promotion_rejects_single_weight_gain() -> None:
    decision = evaluate_no_prior_continuation_promotion(
        _single_weight_gain_rows(),
        [_exposure_row()],
    )

    assert decision["status"] == "not_promotable_no_stable_gain"
    assert decision["benchmark_result"] == "no_prior_continuation_single_weight_gain"


def test_no_prior_continuation_promotion_markdown_reports_limits() -> None:
    markdown = format_no_prior_continuation_promotion_markdown(
        evaluate_no_prior_continuation_promotion(
            _stable_gain_rows(),
            [_exposure_row()],
        )
    )

    assert "# FullMHT No-Prior Continuation Promotion Gate" in markdown
    assert "no_prior_continuation_stable_gain" in markdown
    assert "max_no_prior_continuation_positive_edges_per_subject" in markdown
