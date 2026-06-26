from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
LOCK_PATH = BENCHMARKS / "full_mht_identity_history_method_lock.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runs(filename: str) -> dict[str, dict[str, Any]]:
    return {run["name"]: run for run in _json(BENCHMARKS / filename)["runs"]}


def _normalized(row: dict[str, Any], *, ignored: set[str] | None = None) -> dict[str, Any]:
    skipped = {"name", "output"}
    if ignored:
        skipped.update(ignored)
    return {key: value for key, value in row.items() if key not in skipped}


def _with_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return {**base, **overrides}


def test_identity_history_method_lock_schema_is_current() -> None:
    lock = _json(LOCK_PATH)

    assert lock["schema"] == "full_mht_identity_history_method_lock_v1"


def test_identity_history_central_row_matches_method_lock() -> None:
    lock = _json(LOCK_PATH)
    candidate_runs = _runs("full_mht_identity_history_candidate_manifest.json")

    assert _normalized(candidate_runs["FullMHTIdentityHistory"]) == lock["central_row"]


def test_identity_history_controls_match_method_lock() -> None:
    lock = _json(LOCK_PATH)
    candidate_runs = _runs("full_mht_identity_history_candidate_manifest.json")
    central = dict(lock["central_row"])

    assert _normalized(candidate_runs["FullMHTGreedyIdentityHistory"]) == _with_overrides(
        central,
        lock["greedy_overrides"],
    )
    assert _normalized(
        candidate_runs["FullMHTIdentityHistoryNoLocalContext"],
        ignored=set(lock["local_context_control"]),
    ) == central
    for key, value in lock["local_context_control"].items():
        assert candidate_runs["FullMHTIdentityHistoryNoLocalContext"][key] == value


def test_identity_history_sensitivity_rows_match_method_lock() -> None:
    lock = _json(LOCK_PATH)
    runs = _runs("full_mht_identity_history_sensitivity_manifest.json")
    central = dict(lock["central_row"])

    assert _normalized(runs["IdentityHistoryCentral"]) == central
    for row_name, changed in lock["sensitivity_axes"].items():
        assert _normalized(runs[row_name], ignored=set(changed)) == central
        for key, value in changed.items():
            assert runs[row_name][key] == value


def test_identity_history_addon_rows_match_method_lock() -> None:
    lock = _json(LOCK_PATH)
    central = dict(lock["central_row"])
    scan_runs = _runs("full_mht_identity_history_scan_pruning_manifest.json")
    completion_runs = _runs("full_mht_identity_history_completion_manifest.json")

    assert _normalized(scan_runs["FullMHTIdentityHistory"]) == central
    assert _normalized(completion_runs["FullMHTIdentityHistory"]) == central

    for row_name, weight in lock["scan_pruning_weights"].items():
        row = scan_runs[row_name]
        assert row["scan_motion_history_weight"] == weight
        assert _normalized(row, ignored={"scan_motion_history_weight"}) == central

    for row_name, weight in lock["completion_weights"].items():
        row = completion_runs[row_name]
        assert row["terminal_incomplete_history_weight"] == weight
        assert _normalized(row, ignored={"terminal_incomplete_history_weight"}) == central
