from __future__ import annotations

from bayescatrack.experiments.full_mht_prior_survival_promotion_gate import (
    evaluate_prior_survival_exposure,
    evaluate_prior_survival_promotion,
    evaluate_prior_survival_sensitivity,
    format_prior_survival_promotion_markdown,
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
    survival_complete: float = 0.934,
    greedy_survival_complete: float = 0.931,
) -> list[dict[str, str]]:
    return [
        _metric_row("Track2p", 0.965, 0.924),
        _metric_row("FullMHTPrior2", 0.965, 0.930),
        _metric_row("FullMHTGreedyPrior2", 0.965, 0.924),
        _metric_row("FullMHTPriorVetoScaled", 0.966, 0.932),
        _metric_row("FullMHTGreedyPriorVetoScaled", 0.966, 0.929),
        _metric_row("FullMHTPriorSurvival", 0.966, survival_complete),
        _metric_row("FullMHTGreedyPriorSurvival", 0.966, greedy_survival_complete),
    ]


def _sensitivity_rows(
    *,
    central_complete: float = 0.932,
    weak_neighbors: bool = False,
    pairwise_collapse: bool = False,
) -> list[dict[str, str]]:
    base_pairwise = 0.965
    weak_complete = 0.929 if weak_neighbors else 0.931
    collapsed_pairwise = 0.940 if pairwise_collapse else base_pairwise
    return [
        _metric_row("Track2p", 0.965, 0.924),
        _metric_row("FullMHTPrior2", base_pairwise, 0.930),
        _metric_row("SurvivalW05Clip8", base_pairwise, weak_complete),
        _metric_row("SurvivalW10Clip8", collapsed_pairwise, central_complete),
        _metric_row("SurvivalW15Clip8", base_pairwise, weak_complete),
        _metric_row("SurvivalW10Clip4", base_pairwise, 0.931),
        _metric_row("SurvivalW10MinExamples3", base_pairwise, 0.931),
        _metric_row("SurvivalStrictAnchors", base_pairwise, weak_complete),
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
        "history_prior_survival_weighted_score": "3.0",
        "max_prior_survival_negative_edges_per_subject": "2",
        "max_prior_survival_abs_weighted_score_per_subject": "3.0",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def test_prior_survival_sensitivity_accepts_stable_plateau() -> None:
    sensitivity = evaluate_prior_survival_sensitivity(_sensitivity_rows())

    assert sensitivity["status"] == "complete"
    assert sensitivity["sensitivity_result"] == "stable_plateau"
    assert sensitivity["n_passing_variants"] >= 4
    assert sensitivity["n_passing_weight_variants"] >= 2
    assert "complete_track_f1_macro_delta_vs_base" in sensitivity["deltas"]["SurvivalW10Clip8"]


def test_prior_survival_sensitivity_reports_missing_rows() -> None:
    rows = [row for row in _sensitivity_rows() if row["approach"] != "SurvivalStrictAnchors"]

    sensitivity = evaluate_prior_survival_sensitivity(rows)

    assert sensitivity["status"] == "incomplete"
    assert sensitivity["sensitivity_result"] == "missing_rows"
    assert "SurvivalStrictAnchors" in sensitivity["missing_approaches"]


def test_prior_survival_sensitivity_rejects_knife_edge() -> None:
    sensitivity = evaluate_prior_survival_sensitivity(
        _sensitivity_rows(weak_neighbors=True)
    )

    assert sensitivity["sensitivity_result"] == "sensitivity_not_stable"
    assert sensitivity["n_passing_variants"] < sensitivity["n_required_passing_variants"]


def test_prior_survival_sensitivity_rejects_pairwise_collapse() -> None:
    sensitivity = evaluate_prior_survival_sensitivity(
        _sensitivity_rows(pairwise_collapse=True)
    )

    assert sensitivity["sensitivity_result"] == "pairwise_collapse"
    assert "SurvivalW10Clip8" in sensitivity["pairwise_collapse_variants"]


def test_prior_survival_sensitivity_rejects_macro_pairwise_collapse() -> None:
    rows = _sensitivity_rows()
    for row in rows:
        if row["approach"] == "SurvivalW10Clip8":
            row["pairwise_f1_macro"] = "0.940"

    sensitivity = evaluate_prior_survival_sensitivity(rows)

    assert sensitivity["sensitivity_result"] == "pairwise_collapse"
    assert "SurvivalW10Clip8" in sensitivity["pairwise_collapse_variants"]
    assert sensitivity["deltas"]["SurvivalW10Clip8"]["pairwise_f1_micro_delta_vs_base"] == 0.0
    assert sensitivity["deltas"]["SurvivalW10Clip8"]["pairwise_f1_macro_delta_vs_base"] < -0.01


def test_prior_survival_sensitivity_rejects_macro_complete_regression() -> None:
    rows = _sensitivity_rows()
    for row in rows:
        if row["approach"] == "SurvivalW10Clip8":
            row["complete_track_f1_macro"] = "0.920"

    sensitivity = evaluate_prior_survival_sensitivity(rows)

    assert sensitivity["sensitivity_result"] == "central_candidate_not_stable"
    assert "SurvivalW10Clip8" not in sensitivity["passing_variants"]
    assert sensitivity["deltas"]["SurvivalW10Clip8"]["complete_track_f1_micro_delta_vs_base"] > 0.0
    assert sensitivity["deltas"]["SurvivalW10Clip8"]["complete_track_f1_macro_delta_vs_base"] < 0.0


def test_prior_survival_exposure_requires_survival_columns() -> None:
    stale_row = _exposure_row()
    del stale_row["history_prior_survival_scored_edges"]

    exposure = evaluate_prior_survival_exposure([stale_row])

    assert exposure["status"] == "incomplete"
    assert exposure["exposure_result"] == "missing_prior_survival_exposure_columns"
    assert "history_prior_survival_scored_edges" in exposure["missing_columns"]


def test_prior_survival_exposure_requires_scored_edges() -> None:
    exposure = evaluate_prior_survival_exposure(
        [_exposure_row(history_prior_survival_scored_edges=0)]
    )

    assert exposure["status"] == "incomplete"
    assert exposure["exposure_result"] == "prior_survival_not_scored"


def test_prior_survival_exposure_requires_active_signal() -> None:
    exposure = evaluate_prior_survival_exposure(
        [
            _exposure_row(
                history_prior_survival_scored_edges=12,
                history_prior_survival_positive_edges=0,
                history_prior_survival_negative_edges=0,
            )
        ]
    )

    assert exposure["status"] == "incomplete"
    assert exposure["exposure_result"] == "prior_survival_not_active"


def test_prior_survival_exposure_rejects_broad_negative_penalties() -> None:
    exposure = evaluate_prior_survival_exposure(
        [
            _exposure_row(
                history_prior_survival_negative_edges=12,
                max_prior_survival_negative_edges_per_subject=5,
            )
        ]
    )

    assert exposure["status"] == "complete"
    assert exposure["exposure_result"] == "broad_exposure"
    assert "history_prior_survival_negative_edges" in exposure["failed_limits"]
    assert "max_prior_survival_negative_edges_per_subject" in exposure["failed_limits"]


def test_prior_survival_promotion_requires_all_gates() -> None:
    decision = evaluate_prior_survival_promotion(
        _canonical_rows(),
        _sensitivity_rows(),
        [_exposure_row()],
    )

    assert decision["status"] == "promotable_after_review"
    assert decision["manifest_result"] == "prior_survival_complete_history_advantage"
    assert decision["sensitivity_result"] == "stable_plateau"
    assert decision["exposure_result"] == "bounded_exposure"


def test_prior_survival_promotion_rejects_manifest_tie_against_greedy() -> None:
    decision = evaluate_prior_survival_promotion(
        _canonical_rows(survival_complete=0.931, greedy_survival_complete=0.931),
        _sensitivity_rows(),
        [_exposure_row()],
    )

    assert decision["status"] == "not_promotable_manifest"
    assert decision["manifest_result"] != "prior_survival_complete_history_advantage"


def test_prior_survival_promotion_rejects_unstable_sensitivity() -> None:
    decision = evaluate_prior_survival_promotion(
        _canonical_rows(),
        _sensitivity_rows(weak_neighbors=True),
        [_exposure_row()],
    )

    assert decision["status"] == "not_promotable_sensitivity"
    assert decision["sensitivity_result"] == "sensitivity_not_stable"


def test_prior_survival_promotion_rejects_broad_exposure() -> None:
    decision = evaluate_prior_survival_promotion(
        _canonical_rows(),
        _sensitivity_rows(),
        [_exposure_row(history_selected_non_prior_edges=99)],
    )

    assert decision["status"] == "not_promotable_broad_exposure"
    assert decision["exposure_result"] == "broad_exposure"


def test_prior_survival_promotion_markdown_reports_three_gates() -> None:
    markdown = format_prior_survival_promotion_markdown(
        evaluate_prior_survival_promotion(
            _canonical_rows(),
            _sensitivity_rows(),
            [_exposure_row()],
        )
    )

    assert "# FullMHT Prior-Survival Promotion Gate" in markdown
    assert "prior_survival_complete_history_advantage" in markdown
    assert "stable_plateau" in markdown
    assert "bounded_exposure" in markdown
    assert "history_prior_survival_negative_edges" in markdown
