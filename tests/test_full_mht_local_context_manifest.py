from __future__ import annotations

import json
from pathlib import Path


def _manifest() -> dict:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_local_context_probe_manifest.json"
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_local_context_probe_manifest_is_frozen() -> None:
    manifest = _manifest()
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTLocalContext000",
        "FullMHTLocalContext025",
        "FullMHTLocalContext050",
        "FullMHTLocalContext100",
    ]
    assert runs["FullMHTLocalContext000"]["local_deformation_weight"] == 0.0
    assert runs["FullMHTLocalContext025"]["local_deformation_weight"] == 0.25
    assert runs["FullMHTLocalContext050"]["local_deformation_weight"] == 0.5
    assert runs["FullMHTLocalContext100"]["local_deformation_weight"] == 1.0


def test_local_context_probe_changes_only_local_weight() -> None:
    runs = {run["name"]: run for run in _manifest()["runs"]}
    baseline = runs["FullMHTLocalContext000"]

    shared_keys = [
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
        "track2p_prior_weight",
        "track2p_non_prior_penalty",
        "track2p_prior_switch_penalty",
        "track2p_no_prior_successor_penalty",
        "track2p_prior_miss_penalty",
    ]
    for name in (
        "FullMHTLocalContext025",
        "FullMHTLocalContext050",
        "FullMHTLocalContext100",
    ):
        candidate = runs[name]
        for key in shared_keys:
            assert candidate[key] == baseline[key]


def test_local_context_comparison_includes_all_probe_rows() -> None:
    manifest = _manifest()

    expected = {
        "Track2p",
        "FullMHTLocalContext000",
        "FullMHTLocalContext025",
        "FullMHTLocalContext050",
        "FullMHTLocalContext100",
    }
    for comparison in manifest["comparisons"]:
        assert set(comparison["inputs"]) == expected
