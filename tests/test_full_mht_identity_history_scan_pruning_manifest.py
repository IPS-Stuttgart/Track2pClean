from __future__ import annotations

import json
from pathlib import Path


def _manifest() -> dict:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_identity_history_scan_pruning_manifest.json"
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _without_identity(row: dict) -> dict:
    ignored = {"name", "output", "beam_width", "identity_diverse_beam"}
    return {key: value for key, value in row.items() if key not in ignored}


def test_identity_history_scan_pruning_manifest_is_frozen() -> None:
    manifest = _manifest()
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "FullMHTIdentityHistory",
        "FullMHTGreedyIdentityHistory",
        "IdentityHistoryScanPruning025",
        "GreedyIdentityHistoryScanPruning025",
        "IdentityHistoryScanPruning050",
        "GreedyIdentityHistoryScanPruning050",
        "IdentityHistoryScanPruning100",
        "GreedyIdentityHistoryScanPruning100",
    ]
    assert runs["FullMHTIdentityHistory"]["association_score_mode"] == "calibrated-likelihood"
    assert runs["FullMHTIdentityHistory"]["track2p_prior_survival_weight"] == 1.0
    assert runs["FullMHTIdentityHistory"]["no_prior_continuation_likelihood_weight"] == 1.0
    assert runs["FullMHTIdentityHistory"]["growth_history_prediction_weight"] == 0.5
    assert "scan_motion_history_weight" not in runs["FullMHTIdentityHistory"]
    assert runs["IdentityHistoryScanPruning025"]["scan_motion_history_weight"] == 0.25
    assert runs["IdentityHistoryScanPruning050"]["scan_motion_history_weight"] == 0.5
    assert runs["IdentityHistoryScanPruning100"]["scan_motion_history_weight"] == 1.0


def test_scan_pruning_rows_only_add_scan_history_weight_to_identity_baseline() -> None:
    runs = {run["name"]: run for run in _manifest()["runs"]}
    baseline = runs["FullMHTIdentityHistory"]

    for name, weight in (
        ("IdentityHistoryScanPruning025", 0.25),
        ("IdentityHistoryScanPruning050", 0.5),
        ("IdentityHistoryScanPruning100", 1.0),
    ):
        row = dict(runs[name])
        assert row.pop("scan_motion_history_weight") == weight
        row["name"] = "FullMHTIdentityHistory"
        row["output"] = baseline["output"]
        assert row == baseline


def test_scan_pruning_rows_have_matching_greedy_ablations() -> None:
    runs = {run["name"]: run for run in _manifest()["runs"]}

    for beam_name, greedy_name in (
        ("IdentityHistoryScanPruning025", "GreedyIdentityHistoryScanPruning025"),
        ("IdentityHistoryScanPruning050", "GreedyIdentityHistoryScanPruning050"),
        ("IdentityHistoryScanPruning100", "GreedyIdentityHistoryScanPruning100"),
    ):
        beam = runs[beam_name]
        greedy = runs[greedy_name]
        assert beam["beam_width"] == 8
        assert beam["identity_diverse_beam"] is True
        assert greedy["beam_width"] == 1
        assert greedy["identity_diverse_beam"] is False
        assert _without_identity(greedy) == _without_identity(beam)


def test_scan_pruning_comparison_includes_all_weight_pairs() -> None:
    expected = {
        "Track2p",
        "FullMHTPrior2",
        "FullMHTIdentityHistory",
        "FullMHTGreedyIdentityHistory",
        "IdentityHistoryScanPruning025",
        "GreedyIdentityHistoryScanPruning025",
        "IdentityHistoryScanPruning050",
        "GreedyIdentityHistoryScanPruning050",
        "IdentityHistoryScanPruning100",
        "GreedyIdentityHistoryScanPruning100",
    }

    for comparison in _manifest()["comparisons"]:
        assert set(comparison["inputs"]) == expected
