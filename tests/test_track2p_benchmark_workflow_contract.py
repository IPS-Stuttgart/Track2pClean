from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "track2p-benchmark.yml"
SCRIPT_PATH = PROJECT_ROOT / ".github" / "scripts" / "run_track2p_benchmark.py"


def test_track2p_benchmark_script_is_syntactically_valid() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    ast.parse(source, filename=str(SCRIPT_PATH))


def test_track2p_benchmark_workflow_uploads_results_and_publishes_summary() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "benchmark-results/workflow-summary.md" in workflow
    assert "actions/upload-artifact" in workflow
    assert "track2p-benchmark-results" in workflow
    assert "path: benchmark-results" in workflow


def test_track2p_benchmark_workflow_exposes_regression_gate_inputs() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    expected_gate_names = {
        "TRACK2P_MIN_BEST_PAIRWISE_F1_MACRO",
        "TRACK2P_MIN_BEST_COMPLETE_TRACK_F1_MACRO",
        "TRACK2P_MIN_PAIRWISE_F1_MACRO_DELTA_OVER_BASELINE",
        "TRACK2P_MIN_COMPLETE_TRACK_F1_MACRO_DELTA_OVER_BASELINE",
    }
    for name in expected_gate_names:
        assert name in workflow
        assert name in script


def test_track2p_benchmark_summary_includes_gate_results_and_comparison() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Regression gates" in script
    assert "comparison.md" in script
    assert "workflow-summary.md" in script
