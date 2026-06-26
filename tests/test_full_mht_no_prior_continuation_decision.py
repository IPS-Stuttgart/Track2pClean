from __future__ import annotations

from bayescatrack.experiments.full_mht_no_prior_continuation_decision import (
    evaluate_no_prior_continuation_decision,
    format_no_prior_continuation_markdown,
)


def _metric_row(approach: str, pairwise: float, complete: float) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise),
        "complete_track_f1_micro": str(complete),
        "pairwise_f1_macro": str(pairwise),
        "complete_track_f1_macro": str(complete),
    }


def _base_rows() -> list[dict[str, str]]:
    return [
        _metric_row("Track2p", 0.962, 0.920),
        _metric_row("FullMHTPrior2", 0.965, 0.930),
        _metric_row("FullMHTCalibratedNoDeath", 0.950, 0.850),
    ]


def test_no_prior_continuation_decision_uses_probe_row_names() -> None:
    decision = evaluate_no_prior_continuation_decision(
        [
            *_base_rows(),
            _metric_row("FullMHTNoPriorContinuation050", 0.965, 0.931),
            _metric_row("FullMHTNoPriorContinuation100", 0.965, 0.932),
            _metric_row("FullMHTNoPriorContinuation150", 0.965, 0.930),
        ]
    )

    assert decision["status"] == "complete"
    assert decision["no_prior_continuation_result"] == "no_prior_continuation_stable_gain"
    assert decision["best_candidate"] == "FullMHTNoPriorContinuation100"
    assert decision["permissive_control"] == "FullMHTCalibratedNoDeath"
    assert decision["best_minus_permissive_control_complete_track_f1_micro"] > 0.08
    assert "birth/death likelihood layer" in decision["recommendation"]


def test_no_prior_continuation_decision_reports_missing_rows() -> None:
    decision = evaluate_no_prior_continuation_decision(
        [_metric_row("FullMHTPrior2", 0.965, 0.930)]
    )

    assert decision["status"] == "incomplete"
    assert "Track2p" in decision["missing_approaches"]
    assert "FullMHTCalibratedNoDeath" in decision["missing_approaches"]
    assert "FullMHTNoPriorContinuation050" in decision["missing_approaches"]


def test_no_prior_continuation_decision_flags_single_weight_gain() -> None:
    decision = evaluate_no_prior_continuation_decision(
        [
            *_base_rows(),
            _metric_row("FullMHTNoPriorContinuation050", 0.965, 0.930),
            _metric_row("FullMHTNoPriorContinuation100", 0.965, 0.932),
            _metric_row("FullMHTNoPriorContinuation150", 0.965, 0.930),
        ]
    )

    assert decision["no_prior_continuation_result"] == "no_prior_continuation_single_weight_gain"
    assert decision["recommendation"].startswith("treat as exploratory")


def test_no_prior_continuation_decision_rejects_pairwise_regression() -> None:
    decision = evaluate_no_prior_continuation_decision(
        [
            *_base_rows(),
            _metric_row("FullMHTNoPriorContinuation050", 0.964, 0.933),
            _metric_row("FullMHTNoPriorContinuation100", 0.965, 0.930),
            _metric_row("FullMHTNoPriorContinuation150", 0.965, 0.930),
        ]
    )

    assert decision["no_prior_continuation_result"] == "no_prior_continuation_pairwise_regression"
    assert decision["pairwise_regression_count"] == 1


def test_no_prior_continuation_decision_rejects_complete_regression() -> None:
    decision = evaluate_no_prior_continuation_decision(
        [
            *_base_rows(),
            _metric_row("FullMHTNoPriorContinuation050", 0.965, 0.929),
            _metric_row("FullMHTNoPriorContinuation100", 0.965, 0.930),
            _metric_row("FullMHTNoPriorContinuation150", 0.965, 0.930),
        ]
    )

    assert decision["no_prior_continuation_result"] == "no_prior_continuation_complete_regression"
    assert decision["complete_regression_count"] == 1


def test_no_prior_continuation_decision_markdown_is_specific() -> None:
    markdown = format_no_prior_continuation_markdown(
        evaluate_no_prior_continuation_decision(
            [
                *_base_rows(),
                _metric_row("FullMHTNoPriorContinuation050", 0.965, 0.931),
                _metric_row("FullMHTNoPriorContinuation100", 0.965, 0.932),
                _metric_row("FullMHTNoPriorContinuation150", 0.965, 0.930),
            ]
        )
    )

    assert "# FullMHT No-Prior Continuation Decision" in markdown
    assert "FullMHTNoPriorContinuation100" in markdown
    assert "FullMHTCalibratedNoDeath" in markdown
    assert "no_prior_continuation_stable_gain" in markdown
