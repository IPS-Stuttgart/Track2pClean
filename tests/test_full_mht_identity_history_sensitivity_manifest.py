from __future__ import annotations

import json
from pathlib import Path


def _manifest() -> dict:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_identity_history_sensitivity_manifest.json"
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_identity_history_sensitivity_manifest_is_frozen() -> None:
    manifest = _manifest()
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "IdentityHistorySurvivalW05",
        "IdentityHistoryCentral",
        "IdentityHistorySurvivalW15",
        "IdentityHistoryNoPriorW05",
        "IdentityHistoryNoPriorW15",
        "IdentityHistoryGrowthW025",
        "IdentityHistoryGrowthW100",
    ]
    assert runs["IdentityHistoryCentral"]["association_score_mode"] == "calibrated-likelihood"
    assert runs["IdentityHistoryCentral"]["track2p_prior_survival_weight"] == 1.0
    assert runs["IdentityHistoryCentral"]["no_prior_continuation_likelihood_weight"] == 1.0
    assert runs["IdentityHistoryCentral"]["growth_history_prediction_weight"] == 0.5


def test_identity_history_sensitivity_manifest_changes_one_axis_at_a_time() -> None:
    runs = {run["name"]: run for run in _manifest()["runs"]}
    central = runs["IdentityHistoryCentral"]

    axis_values = {
        "IdentityHistorySurvivalW05": {"track2p_prior_survival_weight": 0.5},
        "IdentityHistorySurvivalW15": {"track2p_prior_survival_weight": 1.5},
        "IdentityHistoryNoPriorW05": {"no_prior_continuation_likelihood_weight": 0.5},
        "IdentityHistoryNoPriorW15": {"no_prior_continuation_likelihood_weight": 1.5},
        "IdentityHistoryGrowthW025": {"growth_history_prediction_weight": 0.25},
        "IdentityHistoryGrowthW100": {"growth_history_prediction_weight": 1.0},
    }
    for name, changed in axis_values.items():
        row = runs[name]
        for key in (
            "association_score_mode",
            "association_likelihood_weight",
            "association_likelihood_clip",
            "track2p_prior_weight",
            "track2p_non_prior_penalty",
            "track2p_prior_switch_penalty",
            "track2p_no_prior_successor_penalty",
            "track2p_prior_miss_penalty",
            "track2p_prior_survival_weight",
            "track2p_prior_survival_min_examples_per_class",
            "track2p_prior_survival_score_clip",
            "no_prior_continuation_likelihood_weight",
            "no_prior_continuation_min_examples_per_class",
            "no_prior_continuation_score_clip",
            "growth_history_prediction_weight",
            "growth_history_prediction_scale",
            "growth_history_prediction_clip",
            "growth_history_prediction_min_edges",
        ):
            if key in changed:
                assert row[key] == changed[key]
            else:
                assert row[key] == central[key]


def test_identity_history_sensitivity_comparison_includes_all_variants() -> None:
    manifest = _manifest()
    expected = {
        "Track2p",
        "FullMHTPrior2",
        "IdentityHistorySurvivalW05",
        "IdentityHistoryCentral",
        "IdentityHistorySurvivalW15",
        "IdentityHistoryNoPriorW05",
        "IdentityHistoryNoPriorW15",
        "IdentityHistoryGrowthW025",
        "IdentityHistoryGrowthW100",
    }

    for comparison in manifest["comparisons"]:
        assert set(comparison["inputs"]) == expected
