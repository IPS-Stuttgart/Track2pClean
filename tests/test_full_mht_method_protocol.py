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
        "test_full_mht_conflict_demo_mht_history_beats_greedy",
        "test_full_mht_conflict_demo_pairwise_good_can_be_complete_bad",
        "test_full_mht_conflict_demo_selection_is_reference_independent",
        "test_full_mht_method_layers_do_not_read_gt_or_audit_columns",
        "test_full_mht_no_gt_leakage_scan_covers_all_method_layers",
        "FullMHTIdentityHistoryNoLocalContext",
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
        "Deterministic edge gating over the same candidates produces the same behavior",
        "The paper text cannot distinguish the benchmark row from post-hoc growth-veto",
    )

    for warning in warnings:
        assert warning in protocol
