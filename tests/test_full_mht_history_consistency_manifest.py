from __future__ import annotations

import json
from pathlib import Path


def _manifest() -> dict:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_history_consistency_probe_manifest.json"
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_history_consistency_probe_manifest_is_frozen() -> None:
    runs = {run["name"]: run for run in _manifest()["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "FullMHTHistoryConsistency050",
        "FullMHTHistoryConsistency100",
        "FullMHTHistoryConsistency200",
        "FullMHTGreedyHistoryConsistency100",
    ]
    assert runs["FullMHTPrior2"]["runner"] == "track2p-full-mht"
    for name, weight in (
        ("FullMHTHistoryConsistency050", 0.5),
        ("FullMHTHistoryConsistency100", 1.0),
        ("FullMHTHistoryConsistency200", 2.0),
    ):
        row = runs[name]
        assert row["runner"] == "track2p-full-mht-history-consistency"
        assert row["history_consistency_weight"] == weight
        assert row["history_consistency_min_history_edges"] == 2
        assert row["history_consistency_joint_margin"] == 1.0
        assert row["history_consistency_score_clip"] == 8.0
        assert row["beam_width"] == runs["FullMHTPrior2"]["beam_width"]
        assert row["scan_hypotheses"] == runs["FullMHTPrior2"]["scan_hypotheses"]
        assert row["track2p_prior_weight"] == runs["FullMHTPrior2"][
            "track2p_prior_weight"
        ]


def test_history_consistency_probe_has_matching_greedy_ablation() -> None:
    runs = {run["name"]: run for run in _manifest()["runs"]}
    candidate = runs["FullMHTHistoryConsistency100"]
    greedy = runs["FullMHTGreedyHistoryConsistency100"]

    assert greedy["runner"] == candidate["runner"]
    assert greedy["beam_width"] == 1
    assert greedy["identity_diverse_beam"] is False
    for key in (
        "seed_source",
        "scan_hypotheses",
        "edge_top_k",
        "miss_cost",
        "full_mht_max_gap",
        "gap_reactivation_cost",
        "min_output_observations",
        "min_edge_score",
        "track2p_prior_weight",
        "track2p_non_prior_penalty",
        "track2p_prior_switch_penalty",
        "track2p_no_prior_successor_penalty",
        "track2p_prior_miss_penalty",
        "history_consistency_weight",
        "history_consistency_min_history_edges",
        "history_consistency_min_feature_scale",
        "history_consistency_joint_margin",
        "history_consistency_score_clip",
    ):
        assert greedy[key] == candidate[key]


def test_history_consistency_probe_comparison_includes_all_rows() -> None:
    manifest = _manifest()

    expected = {
        "Track2p",
        "FullMHTPrior2",
        "FullMHTHistoryConsistency050",
        "FullMHTHistoryConsistency100",
        "FullMHTHistoryConsistency200",
        "FullMHTGreedyHistoryConsistency100",
    }
    for comparison in manifest["comparisons"]:
        assert set(comparison["inputs"]) == expected
        for name in expected:
            assert comparison["inputs"][name] == name
