from __future__ import annotations

from pathlib import Path


def test_full_mht_method_protocol_names_required_method_invariants() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "docs" / "full_mht_method_protocol.md").read_text(encoding="utf-8")
    checklist = (root / "docs" / "full_mht_method_invariant_checklist.md").read_text(
        encoding="utf-8"
    )
    combined = f"{protocol}\n{checklist}"

    required = (
        "test_calibrated_likelihood_flips_scan_assignment_from_local_overlap",
        "test_no_prior_continuation_likelihood_opens_scan_assignment_over_death",
        "test_growth_history_prediction_flips_scan_assignment_to_coherent_history",
        "test_scan_history_conflict_demo_rejects_local_motion_break",
        "test_scan_history_conflict_demo_zero_weight_matches_local_score",
        "test_full_mht_conflict_demo_mht_history_beats_greedy",
        "test_full_mht_conflict_demo_pairwise_good_can_be_complete_bad",
        "test_full_mht_conflict_demo_selection_is_reference_independent",
        "test_full_mht_method_layers_do_not_read_gt_or_audit_columns",
        "test_full_mht_no_gt_leakage_scan_covers_all_method_layers",
        "FullMHTIdentityHistoryNoLocalContext",
        "scan_pruning_stable_complete_history_gain",
        "terminal_completion_stable_gain",
        "bounded_exposure",
        "stable_plateau",
    )

    for needle in required:
        assert needle in combined


def test_full_mht_method_protocol_keeps_non_promotion_warnings() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "docs" / "full_mht_method_protocol.md").read_text(encoding="utf-8")

    warnings = (
        "The frozen identity-history manifest has not been run.",
        "The candidate row ties or loses to its matching greedy ablation.",
        "including the no-local-context control",
        "Scan-history pruning improves only one weight",
        "Deterministic edge gating over the same candidates produces the same behavior",
        "The paper text cannot distinguish the benchmark row from post-hoc growth-veto",
    )

    for warning in warnings:
        assert warning in protocol


def test_full_mht_method_protocol_requires_macro_stable_promotion() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "docs" / "full_mht_method_protocol.md").read_text(encoding="utf-8")

    required = (
        "with no regression in pairwise or complete-track micro/macro F1",
        "on any reported pairwise or complete-track micro/macro metric",
        "with each passing variant non-regressing on all reported micro/macro metrics",
        "regresses on any reported\nmetric",
        "hides a macro-metric\n  regression",
    )

    for needle in required:
        assert needle in protocol


def test_full_mht_method_protocol_supporting_docs_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "docs" / "full_mht_method_protocol.md").read_text(encoding="utf-8")

    required_docs = (
        "docs/full_mht_method_invariant_checklist.md",
        "docs/full_mht_prior_survival_validation.md",
        "docs/full_mht_no_prior_continuation_likelihood.md",
        "docs/full_mht_terminal_completion_objective.md",
        "docs/full_mht_growth_history_prediction.md",
        "docs/full_mht_identity_history_scan_pruning.md",
        "docs/full_mht_label_free_exposure_audit.md",
        "docs/full_mht_manifest_integration_notes.md",
    )

    for doc in required_docs:
        assert doc in protocol
        assert (root / doc).is_file()

    assert "docs/full_mht_growth_history.md" not in protocol
