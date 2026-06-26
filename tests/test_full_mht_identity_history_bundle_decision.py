from __future__ import annotations

from bayescatrack.experiments.full_mht_identity_history_bundle_decision import (
    evaluate_identity_history_bundle,
    format_bundle_markdown,
)


def _identity_promotion(status: str = "promotable_after_review") -> dict[str, str]:
    return {
        "status": status,
        "mht_vs_local_result": "identity_complete_history_advantage",
        "sensitivity_result": "stable_plateau",
        "exposure_result": "bounded_exposure",
    }


def _scan_promotion(status: str = "promotable_after_review") -> dict[str, str]:
    return {
        "status": status,
        "benchmark_result": "scan_pruning_stable_complete_history_gain",
        "exposure_result": "bounded_exposure",
    }


def _terminal_decision(result: str = "terminal_completion_stable_gain") -> dict[str, str]:
    return {
        "status": "complete",
        "terminal_completion_result": result,
    }


def _local_context_decision(result: str = "history_dynamics_stable_gain") -> dict[str, str]:
    return {
        "status": "complete",
        "local_context_result": result,
    }


def test_bundle_promotes_core_when_central_gate_passes() -> None:
    decision = evaluate_identity_history_bundle(_identity_promotion())

    assert decision["status"] == "promotable_core_method"
    assert decision["paper_row"] == "FullMHTIdentityHistory"
    assert decision["core_evidence_result"] == "complete_core_evidence"
    assert decision["optional_variants"] == []
    assert decision["scan_pruning"]["status"] == "not_evaluated"


def test_bundle_records_optional_addons_only_after_core_passes() -> None:
    decision = evaluate_identity_history_bundle(
        _identity_promotion(),
        scan_pruning_promotion=_scan_promotion(),
        terminal_completion=_terminal_decision(),
        local_context=_local_context_decision(),
    )

    assert decision["status"] == "promotable_core_method"
    assert decision["paper_row"] == "FullMHTIdentityHistory"
    assert decision["optional_variants"] == [
        "IdentityHistoryScanPruning",
        "FullMHTIdentityHistoryCompletion",
    ]
    assert decision["local_context"]["status"] == "supporting_component"


def test_bundle_does_not_promote_addons_when_core_fails() -> None:
    decision = evaluate_identity_history_bundle(
        _identity_promotion(status="not_promotable_manifest"),
        scan_pruning_promotion=_scan_promotion(),
        terminal_completion=_terminal_decision(),
    )

    assert decision["status"] == "not_promotable_core_method"
    assert decision["paper_row"] == ""
    assert decision["optional_variants"] == []
    assert decision["scan_pruning"]["status"] == "candidate_addon"
    assert "central FullMHTIdentityHistory gate" in decision["guardrail"]


def test_bundle_rejects_promotable_status_with_inconsistent_core_evidence() -> None:
    promotion = _identity_promotion()
    promotion["exposure_result"] = "broad_exposure"

    decision = evaluate_identity_history_bundle(
        promotion,
        scan_pruning_promotion=_scan_promotion(),
        terminal_completion=_terminal_decision(),
    )

    assert decision["status"] == "not_promotable_core_method"
    assert decision["paper_row"] == ""
    assert decision["core_evidence_result"] == "inconsistent_core_evidence"
    assert decision["optional_variants"] == []
    assert "status and evidence fields disagree" in decision["recommendation"]


def test_bundle_reports_incomplete_core_before_addons() -> None:
    decision = evaluate_identity_history_bundle(
        _identity_promotion(status="incomplete"),
        scan_pruning_promotion=_scan_promotion(),
    )

    assert decision["status"] == "incomplete"
    assert decision["paper_row"] == ""
    assert decision["recommendation"].startswith("rerun the central")


def test_bundle_marks_failed_addons_exploratory() -> None:
    decision = evaluate_identity_history_bundle(
        _identity_promotion(),
        scan_pruning_promotion=_scan_promotion(status="not_promotable_broad_exposure"),
        terminal_completion=_terminal_decision("terminal_completion_single_weight_gain"),
        local_context=_local_context_decision("history_dynamics_single_weight_gain"),
    )

    assert decision["status"] == "promotable_core_method"
    assert decision["optional_variants"] == []
    assert decision["exploratory_variants"] == [
        "IdentityHistoryScanPruning",
        "FullMHTIdentityHistoryCompletion",
    ]
    assert decision["local_context"]["status"] == "exploratory"


def test_bundle_markdown_names_guardrail() -> None:
    markdown = format_bundle_markdown(
        evaluate_identity_history_bundle(
            _identity_promotion(status="not_promotable_manifest"),
            scan_pruning_promotion=_scan_promotion(),
        )
    )

    assert "# FullMHT Identity-History Bundle Decision" in markdown
    assert "not_promotable_core_method" in markdown
    assert "Core evidence" in markdown
    assert "optional add-ons are ignored" in markdown
