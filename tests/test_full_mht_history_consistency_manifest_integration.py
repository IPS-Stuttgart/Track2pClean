from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bayescatrack.experiments import benchmark_manifest as bm
from bayescatrack.experiments import (
    full_mht_history_consistency_manifest_integration as integration,
)
from bayescatrack.experiments import track2p_policy_full_mht_benchmark as base
from bayescatrack.experiments import (
    track2p_policy_full_mht_history_consistency_benchmark as wrapper,
)
from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_full_mht_history_consistency_manifest_aliases_are_supported() -> None:
    assert bm._runner_name("track2p-policy-full-mht-history-consistency") == (
        "track2p-policy-full-mht-history-consistency"
    )
    assert bm._runner_name("track2p-full-mht-history-consistency") == (
        "track2p-policy-full-mht-history-consistency"
    )
    assert bm._runner_name("track2p-pyrecest-full-mht-history-consistency") == (
        "track2p-policy-full-mht-history-consistency"
    )


def test_full_mht_history_consistency_manifest_fields_are_forwarded(tmp_path) -> None:
    manifest_path = tmp_path / "history_consistency_manifest.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data",
                "reference": "reference",
                "reference_kind": "manual-gt",
                "input_format": "suite2p",
                "threshold_method": "min",
                "iou_distance_threshold": 12.0,
                "cell_probability_threshold": 0.5,
            },
            "runs": [
                {
                    "name": "HistoryConsistency",
                    "runner": "track2p-full-mht-history-consistency",
                    "output": "history.csv",
                    "beam_width": 4,
                    "history_consistency_weight": 1.25,
                    "history_consistency_min_history_edges": 3,
                    "history_consistency_joint_margin": 0.75,
                }
            ],
        },
    )

    manifest = load_benchmark_manifest(manifest_path)
    run = manifest.runs[0]

    assert run.runner == "track2p-policy-full-mht-history-consistency"
    assert run.runner_kwargs["threshold_method"] == "min"
    assert run.runner_kwargs["iou_distance_threshold"] == 12.0
    assert run.runner_kwargs["beam_width"] == 4
    assert run.runner_kwargs["history_consistency_weight"] == 1.25
    assert run.runner_kwargs["history_consistency_min_history_edges"] == 3
    assert run.runner_kwargs["history_consistency_joint_margin"] == 0.75


def test_full_mht_history_consistency_manifest_run_uses_patch_context(
    monkeypatch,
) -> None:
    original_method = base.METHOD
    seen: dict[str, object] = {}

    def fake_full_mht_rows(config: object, options: dict) -> list[dict[str, object]]:
        seen["config"] = config
        seen["options"] = dict(options)
        seen["method"] = base.METHOD
        seen["history_weight"] = wrapper._HISTORY_CONFIG.weight
        return [{"subject": "synthetic", "method": base.METHOD}]

    monkeypatch.setattr(
        integration,
        "_run_track2p_policy_full_mht_rows",
        fake_full_mht_rows,
    )

    rows = integration._run_track2p_policy_full_mht_history_consistency_rows(
        SimpleNamespace(),
        {
            "history_consistency_weight": 2.0,
            "history_consistency_min_history_edges": 3,
            "beam_width": 4,
        },
    )

    assert rows == [
        {
            "subject": "synthetic",
            "method": "track2p-policy-full-mht-history-consistency",
        }
    ]
    assert seen["method"] == "track2p-policy-full-mht-history-consistency"
    assert seen["history_weight"] == 2.0
    assert seen["options"] == {
        "history_consistency_weight": 2.0,
        "history_consistency_min_history_edges": 3,
        "beam_width": 4,
    }
    assert base.METHOD == original_method
