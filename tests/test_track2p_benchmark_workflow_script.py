from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_workflow_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "scripts"
        / "run_track2p_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bayescatrack_track2p_workflow_script", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow_script_exposes_shifted_iou_sweep_variants() -> None:
    module = _load_workflow_script()

    shifted_iou_benchmark_runs = getattr(module, "_shifted_iou_benchmark_runs")
    runs_with_labels = shifted_iou_benchmark_runs({"large_cost": 123.0})
    runs = {str(run["name"]): (run, label) for run, label in runs_with_labels}

    assert list(runs) == [
        "global-registered-shifted-iou-r0",
        "global-registered-shifted-iou-r1",
        "global-registered-shifted-iou-r2",
        "global-registered-shifted-iou-r4",
        "global-registered-shifted-iou-r2-p025",
        "global-roi-aware-shifted-r2-p025",
    ]

    r0_kwargs = runs["global-registered-shifted-iou-r0"][0][
        "pairwise_cost_kwargs"
    ]
    assert r0_kwargs["large_cost"] == 123.0
    assert r0_kwargs["shifted_iou_radius"] == 0
    assert r0_kwargs["use_shifted_iou_for_iou_cost"] is False

    r2_kwargs = runs["global-registered-shifted-iou-r2"][0][
        "pairwise_cost_kwargs"
    ]
    assert r2_kwargs["shifted_iou_radius"] == 2
    assert r2_kwargs["use_shifted_iou_for_iou_cost"] is True
    assert r2_kwargs["shifted_iou_shift_penalty_weight"] == 0.0

    penalized_kwargs = runs["global-registered-shifted-iou-r2-p025"][0][
        "pairwise_cost_kwargs"
    ]
    assert penalized_kwargs["shifted_iou_shift_penalty_weight"] == 0.25

    roi_run, roi_label = runs["global-roi-aware-shifted-r2-p025"]
    assert roi_label == "Global ROI-aware shifted r=2 p=0.25"
    assert roi_run["cost"] == "roi-aware-shifted"
    assert roi_run["pairwise_cost_kwargs"][
        "use_shifted_mask_cosine_for_mask_cosine_cost"
    ] is True
