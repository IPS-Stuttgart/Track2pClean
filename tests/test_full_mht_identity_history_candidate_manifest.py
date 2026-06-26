from __future__ import annotations

import json
from pathlib import Path


def _manifest() -> dict:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_identity_history_candidate_manifest.json"
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_identity_history_candidate_manifest_is_frozen() -> None:
    manifest = _manifest()
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "FullMHTPriorSurvival",
        "FullMHTNoPriorContinuation100",
        "FullMHTIdentityHistory",
        "FullMHTGreedyIdentityHistory",
    ]
    assert runs["FullMHTIdentityHistory"]["runner"] == "track2p-full-mht"
    assert runs["FullMHTIdentityHistory"]["association_score_mode"] == (
        "calibrated-likelihood"
    )
    assert runs["FullMHTIdentityHistory"]["track2p_prior_survival_weight"] == 1.0
    assert runs["FullMHTIdentityHistory"]["no_prior_continuation_likelihood_weight"] == 1.0
    assert runs["FullMHTIdentityHistory"]["growth_history_prediction_weight"] == 0.5


def test_identity_history_candidate_has_matching_greedy_ablation() -> None:
    runs = {run["name"]: run for run in _manifest()["runs"]}
    candidate = runs["FullMHTIdentityHistory"]
    greedy = runs["FullMHTGreedyIdentityHistory"]

    assert greedy["beam_width"] == 1
    assert greedy["identity_diverse_beam"] is False
    for key in (
        "runner",
        "seed_source",
        "scan_hypotheses",
        "edge_top_k",
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
        assert greedy[key] == candidate[key]


def test_identity_history_comparison_includes_candidate_and_greedy() -> None:
    manifest = _manifest()

    for comparison in manifest["comparisons"]:
        assert comparison["inputs"]["FullMHTIdentityHistory"] == "FullMHTIdentityHistory"
        assert comparison["inputs"]["FullMHTGreedyIdentityHistory"] == (
            "FullMHTGreedyIdentityHistory"
        )
