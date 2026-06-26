from __future__ import annotations

from bayescatrack.experiments.full_mht_scan_history_dynamics_decision import (
    evaluate_scan_history_dynamics_decision,
)


def _metric_row(approach: str, pairwise: float, complete: float) -> dict[str, str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": str(pairwise),
        "complete_track_f1_micro": str(complete),
        "pairwise_f1_macro": str(pairwise),
        "complete_track_f1_macro": str(complete),
    }


def test_scan_history_dynamics_decision_uses_scan_probe_row_names() -> None:
    decision = evaluate_scan_history_dynamics_decision(
        [
            _metric_row("Track2p", 0.962, 0.920),
            _metric_row("FullMHTPrior2", 0.965, 0.930),
            _metric_row("FullMHTScanHistoryDynamics025", 0.965, 0.931),
            _metric_row("FullMHTScanHistoryDynamics050", 0.965, 0.932),
            _metric_row("FullMHTScanHistoryDynamics100", 0.965, 0.930),
        ]
    )

    assert decision["status"] == "complete"
    assert decision["history_dynamics_result"] == "history_dynamics_stable_gain"
    assert decision["best_candidate"] == "FullMHTScanHistoryDynamics050"
