from __future__ import annotations

from pathlib import Path

METHOD_LAYER_MODULES = (
    "src/bayescatrack/experiments/_full_mht_manifest_integration.py",
    "src/bayescatrack/experiments/full_mht_history_dynamics_integration.py",
    "src/bayescatrack/experiments/full_mht_scan_history_dynamics_integration.py",
    "src/bayescatrack/experiments/full_mht_scan_history_conflict_demo.py",
    "src/bayescatrack/experiments/full_mht_growth_history_prediction_integration.py",
    "src/bayescatrack/experiments/full_mht_growth_history_prediction_promotion_gate.py",
    "src/bayescatrack/experiments/full_mht_local_context_integration.py",
    "src/bayescatrack/experiments/full_mht_local_context_decision.py",
    "src/bayescatrack/experiments/full_mht_no_prior_continuation_model.py",
    "src/bayescatrack/experiments/full_mht_no_prior_continuation_integration.py",
    "src/bayescatrack/experiments/full_mht_no_prior_continuation_manifest_integration.py",
    "src/bayescatrack/experiments/full_mht_no_prior_continuation_decision.py",
    "src/bayescatrack/experiments/full_mht_no_prior_continuation_promotion_gate.py",
    "src/bayescatrack/experiments/full_mht_terminal_completion_integration.py",
    "src/bayescatrack/experiments/full_mht_prior_survival_model.py",
    "src/bayescatrack/experiments/full_mht_prior_survival_integration.py",
    "src/bayescatrack/experiments/full_mht_prior_survival_promotion_gate.py",
    "src/bayescatrack/experiments/full_mht_identity_history_decision.py",
    "src/bayescatrack/experiments/full_mht_identity_history_promotion_gate.py",
    "src/bayescatrack/experiments/track2p_policy_full_mht_exposure_audit.py",
)

FORBIDDEN_GT_TOKENS = (
    "edge_status_against_gt",
    "pairwise_delta_if_removed",
    "complete_delta_if_removed",
    "reference_identity",
    "manual_gt_status",
    "ground_truth_reference_source",
    "_load_reference_for_subject",
    "_score_prediction_against_reference",
)


def test_full_mht_method_layers_do_not_read_gt_or_audit_columns() -> None:
    root = Path(__file__).resolve().parents[1]

    for relative_path in METHOD_LAYER_MODULES:
        text = (root / relative_path).read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_GT_TOKENS:
            assert token not in text, f"{relative_path} contains forbidden token {token!r}"


def test_full_mht_no_gt_leakage_scan_covers_all_integration_layers() -> None:
    root = Path(__file__).resolve().parents[1]
    experiments = root / "src" / "bayescatrack" / "experiments"
    expected = {
        str(path.relative_to(root))
        for pattern in (
            "_full_mht_manifest_integration.py",
            "full_mht_*_integration.py",
            "full_mht_*_model.py",
            "track2p_policy_full_mht_exposure_audit.py",
        )
        for path in experiments.glob(pattern)
    }

    assert expected <= set(METHOD_LAYER_MODULES)
