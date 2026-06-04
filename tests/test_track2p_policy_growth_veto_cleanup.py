from __future__ import annotations

from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup


def _candidate_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "session_a": 5,
        "session_b": 6,
        "roi_a": 2309,
        "roi_b": 1210,
        "occurrence_index": 0,
        "remove_reason": "split_edge",
        "would_split_component": 1,
        "is_terminal_edge": 1,
        "is_last_session_edge": 1,
        "complete_component_size": 7,
        "growth_anchor_count": 2,
        "growth_residual_mahalanobis": 26.0,
        "registered_iou": 0.55,
        "shifted_iou": 0.76,
        "cell_probability_a": 0.57,
        "cell_probability_b": 0.81,
    }
    row.update(overrides)
    return row


def test_growth_veto_cleanup_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-growth-veto-cleanup"]

    assert canonical == "track2p-policy-growth-veto-cleanup"
    assert cli._BENCHMARK_ALIASES["track2p-component-growth-veto-cleanup"] == canonical
    assert (
        cli._BENCHMARK_COMMANDS[canonical].module
        == "bayescatrack.experiments.track2p_policy_growth_veto_cleanup"
    )


def test_growth_veto_gate_accepts_extreme_terminal_complete_edge() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(), cleanup.GrowthVetoGate(), n_sessions=7
    )

    assert reason == "accepted"


def test_growth_veto_gate_rejects_nonterminal_edges_by_default() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(is_terminal_edge=0), cleanup.GrowthVetoGate(), n_sessions=7
    )

    assert reason == "not_terminal_edge"


def test_growth_veto_gate_rejects_incomplete_components_by_default() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(complete_component_size=6), cleanup.GrowthVetoGate(), n_sessions=7
    )

    assert reason == "not_complete_component"


def test_growth_veto_gate_rejects_low_growth_residual() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(growth_residual_mahalanobis=12.0),
        cleanup.GrowthVetoGate(),
        n_sessions=7,
    )

    assert reason == "growth_residual_mahalanobis_below_gate"


def test_growth_veto_gate_rejects_low_cell_probability() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(cell_probability_a=0.49), cleanup.GrowthVetoGate(), n_sessions=7
    )

    assert reason == "cell_probability_below_gate"


def test_growth_veto_gate_rejects_low_anchor_count() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(growth_anchor_count=1),
        cleanup.GrowthVetoGate(min_anchor_count=2),
        n_sessions=7,
    )

    assert reason == "growth_anchor_count_below_gate"


def test_growth_veto_gate_rejects_small_complete_component_when_requested() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(complete_component_size=6),
        cleanup.GrowthVetoGate(
            require_complete_component=False,
            min_complete_component_size=7,
        ),
        n_sessions=7,
    )

    assert reason == "complete_component_size_below_gate"


def test_growth_veto_row_selection_respects_per_subject_cap() -> None:
    rows = [
        _candidate_row(growth_residual_mahalanobis=26.0, roi_b=1210),
        _candidate_row(growth_residual_mahalanobis=30.0, roi_b=1211),
    ]

    selected = cleanup._selected_growth_veto_rows(
        rows, gate=cleanup.GrowthVetoGate(max_vetoes_per_subject=1), n_sessions=7
    )

    assert len(selected) == 1
    assert selected[0]["roi_b"] == 1211
