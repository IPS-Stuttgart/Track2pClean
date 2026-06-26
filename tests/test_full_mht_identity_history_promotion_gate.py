from __future__ import annotations

from bayescatrack.experiments.full_mht_identity_history_promotion_gate import (
    evaluate_identity_history_exposure,
    evaluate_identity_history_promotion,
    evaluate_identity_history_sensitivity,
    format_identity_history_promotion_markdown,
)


def _metric_row(approach: str, pairwise: float, complete: float) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise),
        "complete_track_f1_micro": str(complete),
        "pairwise_f1_macro": str(pairwise),
        "complete_track_f1_macro": str(complete),
    }


def _canonical_rows(
    *,
    identity_pairwise: float = 0.966,
    identity_complete: float = 0.934,
    greedy_pairwise: float = 0.966,
    greedy_complete: float = 0.931,
    no_local_pairwise: float = 0.965,
    no_local_complete: float = 0.932,
) -> list[dict[str, str]]:
    return [
        _metric_row("Track2p", 0.965, 0.924),
        _metric_row("FullMHTPrior2", 0.965, 0.930),
        _metric_row("FullMHTPriorSurvival", 0.966, 0.932),
        _metric_row("FullMHTNoPriorContinuation100", 0.965, 0.930),
        _metric_row("FullMHTIdentityHistoryNoLocalContext", no_local_pairwise, no_local_complete),
        _metric_row("FullMHTIdentityHistory", identity_pairwise, identity_complete),
        _metric_row("FullMHTGreedyIdentityHistory", greedy_pairwise, greedy_complete),
    ]


def _sensitivity_rows(
    *,
    central_pairwise: float = 0.965,
    central_complete: float = 0.932,
    weak_neighbors: bool = False,
) -> list[dict[str, str]]:
    neighbor_complete = 0.929 if weak_neighbors else 0.932
    return [
        _metric_row("Track2p", 0.965, 0.924),
        _metric_row("FullMHTPrior2", 0.965, 0.930),
        _metric_row("IdentityHistorySurvivalW05", 0.965, neighbor_complete),
        _metric_row("IdentityHistoryCentral", central_pairwise, central_complete),
        _metric_row("IdentityHistorySurvivalW15", 0.965, neighbor_complete),
        _metric_row("IdentityHistoryNoPriorW05", 0.965, neighbor_complete),
        _metric_row("IdentityHistoryNoPriorW15", 0.965, neighbor_complete),
        _metric_row("IdentityHistoryGrowthW025", 0.965, neighbor_complete),
        _metric_row("IdentityHistoryGrowthW100", 0.965, neighbor_complete),
    ]


def _exposure_row(**overrides: object) -> dict[str, str]:
    row = {
        "subject": "ALL",
        "max_selected_non_prior_edges_per_subject": "2",
        "history_selected_non_prior_edges": "4",
        "history_switched_prior_successors": "0",
        "history_no_prior_successor_continuations": "4",
        "history_gap_reactivated_tracks": "0",
        "history_prior_survival_scored_edges": "12",
        "history_prior_survival_positive_edges": "10",
        "history_prior_survival_negative_edges": "2",
        "max_prior_survival_negative_edges_per_subject": "2",
        "history_no_prior_continuation_scored_edges": "4",
        "history_no_prior_continuation_positive_edges": "2",
        "history_no_prior_continuation_negative_edges": "2",
        "max_no_prior_continuation_positive_edges_per_subject": "2",
        "max_no_prior_continuation_abs_weighted_score_per_subject": "2.0",
        "history_growth_prediction_evaluated_edges": "6",
        "history_growth_prediction_penalized_edges": "2",
        "max_growth_prediction_penalized_edges_per_subject": "2",
        "max_growth_prediction_weighted_penalty_per_subject": "1.5",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def test_identity_history_sensitivity_accepts_stable_plateau() -> None:
    sensitivity = evaluate_identity_history_sensitivity(_sensitivity_rows())

    assert sensitivity["status"] == "complete"
    assert sensitivity["sensitivity_result"] == "stable_plateau"
    assert sensitivity["n_passing_variants"] >= 5
    assert sensitivity["axis_passing_counts"] == {
        "survival": 3,
        "no_prior": 3,
        "growth": 3,
    }
    assert "complete_track_f1_macro_delta_vs_base" in sensitivity["deltas"]["IdentityHistoryCentral"]


def test_identity_history_sensitivity_reports_missing_rows() -> None:
    rows = [row for row in _sensitivity_rows() if row["approach"] != "IdentityHistoryGrowthW100"]

    sensitivity = evaluate_identity_history_sensitivity(rows)

    assert sensitivity["status"] == "incomplete"
    assert sensitivity["sensitivity_result"] == "missing_rows"
    assert "IdentityHistoryGrowthW100" in sensitivity["missing_approaches"]


def test_identity_history_sensitivity_rejects_knife_edge() -> None:
    sensitivity = evaluate_identity_history_sensitivity(_sensitivity_rows(weak_neighbors=True))

    assert sensitivity["sensitivity_result"] == "sensitivity_not_stable"
    assert sensitivity["n_passing_variants"] < sensitivity["n_required_passing_variants"]


def test_identity_history_sensitivity_rejects_pairwise_collapse() -> None:
    sensitivity = evaluate_identity_history_sensitivity(_sensitivity_rows(central_pairwise=0.950))

    assert sensitivity["sensitivity_result"] == "pairwise_collapse"
    assert "IdentityHistoryCentral" in sensitivity["pairwise_collapse_variants"]


def test_identity_history_sensitivity_rejects_macro_complete_regression() -> None:
    rows = _sensitivity_rows()
    for row in rows:
        if row["approach"] == "IdentityHistoryCentral":
            row["complete_track_f1_macro"] = "0.920"

    sensitivity = evaluate_identity_history_sensitivity(rows)

    assert sensitivity["sensitivity_result"] == "central_candidate_not_stable"
    assert "IdentityHistoryCentral" not in sensitivity["passing_variants"]
    assert sensitivity["deltas"]["IdentityHistoryCentral"]["complete_track_f1_macro_delta_vs_base"] < 0.0


def test_identity_history_exposure_requires_identity_columns() -> None:
    stale_row = _exposure_row()
    del stale_row["history_growth_prediction_evaluated_edges"]

    exposure = evaluate_identity_history_exposure([stale_row])

    assert exposure["status"] == "incomplete"
    assert exposure["exposure_result"] == "missing_identity_history_exposure_columns"
    assert "history_growth_prediction_evaluated_edges" in exposure["missing_columns"]


def test_identity_history_exposure_requires_active_layer_signals() -> None:
    exposure = evaluate_identity_history_exposure(
        [_exposure_row(history_no_prior_continuation_positive_edges=0, history_no_prior_continuation_negative_edges=0)]
    )

    assert exposure["status"] == "incomplete"
    assert exposure["exposure_result"] == "identity_history_layers_not_active"
    assert "no_prior_continuation" in exposure["inactive_layers"]


def test_identity_history_exposure_rejects_broad_layer_firing() -> None:
    exposure = evaluate_identity_history_exposure(
        [
            _exposure_row(
                history_growth_prediction_penalized_edges=12,
                max_growth_prediction_penalized_edges_per_subject=5,
            )
        ]
    )

    assert exposure["status"] == "complete"
    assert exposure["exposure_result"] == "broad_exposure"
    assert "history_growth_prediction_penalized_edges" in exposure["failed_limits"]
    assert "max_growth_prediction_penalized_edges_per_subject" in exposure["failed_limits"]


def test_identity_history_promotion_requires_all_gates() -> None:
    decision = evaluate_identity_history_promotion(
        _canonical_rows(),
        _sensitivity_rows(),
        [_exposure_row()],
    )

    assert decision["status"] == "promotable_after_review"
    assert decision["mht_vs_local_result"] == "identity_complete_history_advantage"
    assert decision["history_search_result"] == "identity_complete_history_advantage"
    assert decision["no_local_context_control_result"] == "identity_improves_no_local_context"
    assert decision["sensitivity_result"] == "stable_plateau"
    assert decision["exposure_result"] == "bounded_exposure"


def test_identity_history_promotion_rejects_manifest_tie_against_greedy() -> None:
    decision = evaluate_identity_history_promotion(
        _canonical_rows(identity_complete=0.931, greedy_complete=0.931),
        _sensitivity_rows(),
        [_exposure_row()],
    )

    assert decision["status"] == "not_promotable_manifest"
    assert decision["mht_vs_local_result"] != "identity_complete_history_advantage"


def test_identity_history_promotion_rejects_manifest_macro_regression() -> None:
    rows = _canonical_rows(identity_complete=0.934, greedy_complete=0.931)
    for row in rows:
        if row["approach"] == "FullMHTIdentityHistory":
            row["complete_track_f1_macro"] = "0.900"

    decision = evaluate_identity_history_promotion(
        rows,
        _sensitivity_rows(),
        [_exposure_row()],
    )

    assert decision["status"] == "not_promotable_manifest"
    assert decision["mht_vs_local_result"] == "identity_regression_vs_greedy"


def test_identity_history_promotion_rejects_no_local_context_regression() -> None:
    decision = evaluate_identity_history_promotion(
        _canonical_rows(no_local_pairwise=0.966, no_local_complete=0.936),
        _sensitivity_rows(),
        [_exposure_row()],
    )

    assert decision["status"] == "not_promotable_manifest"
    assert decision["no_local_context_control_result"] == "identity_below_no_local_context"


def test_identity_history_promotion_rejects_unstable_sensitivity() -> None:
    decision = evaluate_identity_history_promotion(
        _canonical_rows(),
        _sensitivity_rows(weak_neighbors=True),
        [_exposure_row()],
    )

    assert decision["status"] == "not_promotable_sensitivity"
    assert decision["sensitivity_result"] == "sensitivity_not_stable"


def test_identity_history_promotion_rejects_broad_exposure() -> None:
    decision = evaluate_identity_history_promotion(
        _canonical_rows(),
        _sensitivity_rows(),
        [_exposure_row(history_selected_non_prior_edges=99)],
    )

    assert decision["status"] == "not_promotable_broad_exposure"
    assert decision["exposure_result"] == "broad_exposure"


def test_identity_history_promotion_markdown_reports_three_gates() -> None:
    markdown = format_identity_history_promotion_markdown(
        evaluate_identity_history_promotion(
            _canonical_rows(),
            _sensitivity_rows(),
            [_exposure_row()],
        )
    )

    assert "# FullMHT Identity-History Promotion Gate" in markdown
    assert "MHT-vs-local result" in markdown
    assert "No-local-context result" in markdown
    assert "identity_complete_history_advantage" in markdown
    assert "stable_plateau" in markdown
    assert "bounded_exposure" in markdown
    assert "history_growth_prediction_penalized_edges" in markdown
