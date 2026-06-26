from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def _manifest(filename: str) -> dict[str, Any]:
    return json.loads((BENCHMARKS / filename).read_text(encoding="utf-8"))


def _runs(filename: str) -> dict[str, dict[str, Any]]:
    return {run["name"]: run for run in _manifest(filename)["runs"]}


def _normalized(row: dict[str, Any], *, extra_ignored: set[str] | None = None) -> dict[str, Any]:
    ignored = {"name", "output"}
    if extra_ignored:
        ignored.update(extra_ignored)
    return {key: value for key, value in row.items() if key not in ignored}


def test_identity_history_central_row_is_shared_across_manifests() -> None:
    candidate = _runs("full_mht_identity_history_candidate_manifest.json")[
        "FullMHTIdentityHistory"
    ]
    sensitivity = _runs("full_mht_identity_history_sensitivity_manifest.json")[
        "IdentityHistoryCentral"
    ]
    scan_pruning = _runs("full_mht_identity_history_scan_pruning_manifest.json")[
        "FullMHTIdentityHistory"
    ]
    completion = _runs("full_mht_identity_history_completion_manifest.json")[
        "FullMHTIdentityHistory"
    ]

    expected = _normalized(candidate)
    assert _normalized(sensitivity) == expected
    assert _normalized(scan_pruning) == expected
    assert _normalized(completion) == expected


def test_identity_history_greedy_ablation_is_shared_where_used() -> None:
    candidate = _runs("full_mht_identity_history_candidate_manifest.json")[
        "FullMHTGreedyIdentityHistory"
    ]
    scan_pruning = _runs("full_mht_identity_history_scan_pruning_manifest.json")[
        "FullMHTGreedyIdentityHistory"
    ]

    assert _normalized(scan_pruning) == _normalized(candidate)
    assert candidate["beam_width"] == 1
    assert candidate["identity_diverse_beam"] is False


def test_scan_pruning_addon_rows_change_only_scan_history_weight() -> None:
    baseline = _normalized(
        _runs("full_mht_identity_history_candidate_manifest.json")["FullMHTIdentityHistory"]
    )
    scan_runs = _runs("full_mht_identity_history_scan_pruning_manifest.json")
    expected_weights = {
        "IdentityHistoryScanPruning025": 0.25,
        "IdentityHistoryScanPruning050": 0.50,
        "IdentityHistoryScanPruning100": 1.00,
    }

    for row_name, weight in expected_weights.items():
        row = scan_runs[row_name]
        assert row["scan_motion_history_weight"] == weight
        assert _normalized(row, extra_ignored={"scan_motion_history_weight"}) == baseline


def test_completion_addon_rows_change_only_terminal_weight() -> None:
    baseline = _normalized(
        _runs("full_mht_identity_history_candidate_manifest.json")["FullMHTIdentityHistory"]
    )
    completion_runs = _runs("full_mht_identity_history_completion_manifest.json")
    expected_weights = {
        "FullMHTIdentityHistoryCompletion025": 0.25,
        "FullMHTIdentityHistoryCompletion050": 0.50,
        "FullMHTIdentityHistoryCompletion100": 1.00,
    }

    for row_name, weight in expected_weights.items():
        row = completion_runs[row_name]
        assert row["terminal_incomplete_history_weight"] == weight
        assert _normalized(row, extra_ignored={"terminal_incomplete_history_weight"}) == baseline


def test_sensitivity_rows_change_only_one_declared_axis() -> None:
    baseline = _normalized(
        _runs("full_mht_identity_history_candidate_manifest.json")["FullMHTIdentityHistory"]
    )
    sensitivity_runs = _runs("full_mht_identity_history_sensitivity_manifest.json")
    expected_changes = {
        "IdentityHistorySurvivalW05": {"track2p_prior_survival_weight": 0.5},
        "IdentityHistorySurvivalW15": {"track2p_prior_survival_weight": 1.5},
        "IdentityHistoryNoPriorW05": {"no_prior_continuation_likelihood_weight": 0.5},
        "IdentityHistoryNoPriorW15": {"no_prior_continuation_likelihood_weight": 1.5},
        "IdentityHistoryGrowthW025": {"growth_history_prediction_weight": 0.25},
        "IdentityHistoryGrowthW100": {"growth_history_prediction_weight": 1.0},
    }

    for row_name, changed in expected_changes.items():
        row = sensitivity_runs[row_name]
        for key, value in changed.items():
            assert row[key] == value
        assert _normalized(row, extra_ignored=set(changed)) == baseline
