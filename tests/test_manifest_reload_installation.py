from __future__ import annotations

import importlib


def test_manifest_rows_are_restored_after_workbench_reload():
    integration = importlib.import_module(
        "bayescatrack.experiments._teacher_" + "rescue_manifest_integration"
    )
    workbench = importlib.import_module(
        "bayescatrack.experiments.advanced_improvement_workbench"
    )

    reloaded = importlib.reload(workbench)
    integration.install_teacher_rescue_manifest_integration()
    integration.install_teacher_rescue_manifest_integration()

    manifest = reloaded.track2p_result_improvement_manifest(
        data_root="data",
        reference_root="reference",
        output_root="results",
    )

    names = [run["name"] for run in manifest["runs"]]
    row_name = (
        "track2p-policy-teacher-adjacent-" "rescue-dynamic-confidence-seed-source"
    )
    assert row_name in names
    assert names.count(row_name) == 1
