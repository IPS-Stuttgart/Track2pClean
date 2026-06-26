from __future__ import annotations

import json
from pathlib import Path


def _manifest() -> dict:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_identity_history_completion_manifest.json"
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_identity_history_completion_manifest_is_frozen() -> None:
    manifest = _manifest()
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTIdentityHistory",
        "FullMHTIdentityHistoryCompletion025",
        "FullMHTIdentityHistoryCompletion050",
        "FullMHTIdentityHistoryCompletion100",
    ]
    assert runs["FullMHTIdentityHistory"]["association_score_mode"] == "calibrated-likelihood"
    assert runs["FullMHTIdentityHistory"]["track2p_prior_survival_weight"] == 1.0
    assert runs["FullMHTIdentityHistory"]["no_prior_continuation_likelihood_weight"] == 1.0
    assert runs["FullMHTIdentityHistory"]["growth_history_prediction_weight"] == 0.5


def test_identity_history_completion_manifest_changes_only_terminal_weight() -> None:
    runs = {run["name"]: run for run in _manifest()["runs"]}
    baseline = runs["FullMHTIdentityHistory"]
    expected_weights = {
        "FullMHTIdentityHistoryCompletion025": 0.25,
        "FullMHTIdentityHistoryCompletion050": 0.50,
        "FullMHTIdentityHistoryCompletion100": 1.00,
    }

    for name, weight in expected_weights.items():
        row = runs[name]
        assert row["terminal_incomplete_history_weight"] == weight
        for key in (
            "runner",
            "seed_source",
            "beam_width",
            "scan_hypotheses",
            "edge_top_k",
            "identity_diverse_beam",
            "miss_cost",
            "full_mht_max_gap",
            "gap_reactivation_cost",
            "min_output_observations",
            "min_edge_score",
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
            assert row[key] == baseline[key]


def test_identity_history_completion_comparison_includes_all_variants() -> None:
    manifest = _manifest()
    expected = {
        "Track2p",
        "FullMHTIdentityHistory",
        "FullMHTIdentityHistoryCompletion025",
        "FullMHTIdentityHistoryCompletion050",
        "FullMHTIdentityHistoryCompletion100",
    }

    for comparison in manifest["comparisons"]:
        assert set(comparison["inputs"]) == expected
