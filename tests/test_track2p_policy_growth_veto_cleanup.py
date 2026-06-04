from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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
        "edge_source": "policy",
        "would_split_component": 1,
        "is_terminal_edge": 1,
        "is_last_session_edge": 1,
        "complete_component_size": 7,
        "growth_anchor_count": 2,
        "growth_residual_mahalanobis": 26.0,
        "registered_iou": 0.55,
        "shifted_iou": 0.76,
        "row_rank": 1,
        "column_rank": 1,
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


def test_growth_veto_cleanup_parser_exposes_conservative_defaults() -> None:
    args = cleanup.build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "growth_veto_cleanup.csv",
        ]
    )

    assert args.min_growth_residual_mahalanobis == 20.0
    assert args.min_veto_registered_iou == 0.45
    assert args.min_veto_shifted_iou == 0.60
    assert args.max_veto_registered_iou is None
    assert args.max_veto_shifted_iou is None
    assert args.min_veto_cell_probability == 0.50
    assert args.max_veto_row_rank == 1
    assert args.max_veto_column_rank == 1
    assert args.require_veto_not_suffix_edge is True
    assert args.max_vetoes_per_subject == 1


def test_growth_veto_gate_accepts_extreme_terminal_complete_edge() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(), cleanup.GrowthVetoGate(), n_sessions=7
    )

    assert reason == "accepted"


def test_growth_veto_gate_rejects_coherence_suffix_edges_by_default() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(edge_source="suffix"), cleanup.GrowthVetoGate(), n_sessions=7
    )

    assert reason == "coherence_suffix_edge"


def test_growth_veto_gate_rejects_nonterminal_edges_by_default() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(is_terminal_edge=0), cleanup.GrowthVetoGate(), n_sessions=7
    )

    assert reason == "not_terminal_edge"


def test_growth_veto_gate_rejects_incomplete_components_by_default() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(complete_component_size=6),
        cleanup.GrowthVetoGate(),
        n_sessions=7,
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


def test_growth_veto_gate_rejects_non_top_rank_edges() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(row_rank=2), cleanup.GrowthVetoGate(), n_sessions=7
    )

    assert reason == "row_rank_above_gate"


def test_growth_veto_gate_rejects_too_strong_registered_iou_when_capped() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(registered_iou=0.68),
        cleanup.GrowthVetoGate(max_registered_iou=0.60),
        n_sessions=7,
    )

    assert reason == "registered_iou_above_gate"


def test_growth_veto_gate_rejects_too_strong_shifted_iou_when_capped() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(shifted_iou=0.85),
        cleanup.GrowthVetoGate(max_shifted_iou=0.80),
        n_sessions=7,
    )

    assert reason == "shifted_iou_above_gate"


def test_growth_veto_sparse_shifted_iou_fill_only_touches_prequalified_rows(
    monkeypatch,
) -> None:
    rows = [
        _candidate_row(session_a=0, session_b=1, shifted_iou=float("nan"), roi_b=1210),
        _candidate_row(
            session_a=0,
            session_b=1,
            shifted_iou=float("nan"),
            roi_b=1211,
            growth_residual_mahalanobis=5.0,
        ),
    ]
    sessions = (_fake_session([2309], 0), _fake_session([1210, 1211], 1))

    monkeypatch.setattr(cleanup, "_roi_indices", lambda session: session.roi_indices)
    monkeypatch.setattr(
        cleanup,
        "register_plane_pair",
        lambda _reference, moving, *, transform_type: moving,
    )
    monkeypatch.setattr(
        cleanup,
        "_pairwise_shifted_iou_from_support",
        lambda _reference, measurement, *, radius: {
            "shifted_iou": np.asarray([[0.76, 0.20]], dtype=float)[
                :, : measurement.shape[0]
            ]
        },
    )

    augmented = cleanup._augment_growth_veto_candidate_shifted_iou(  # pylint: disable=protected-access
        rows,
        sessions,
        gate=cleanup.GrowthVetoGate(),
        n_sessions=2,
    )

    assert augmented[0]["shifted_iou"] == 0.76
    assert augmented[1]["shifted_iou"] != augmented[1]["shifted_iou"]


def test_growth_veto_sparse_shifted_iou_fill_respects_registered_iou_cap(
    monkeypatch,
) -> None:
    rows = [
        _candidate_row(shifted_iou=float("nan"), registered_iou=0.70),
    ]
    sessions = (_fake_session([2309], 0), _fake_session([1210], 1))

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("shifted IoU should not be computed for capped edges")

    monkeypatch.setattr(cleanup, "register_plane_pair", fail_if_called)

    augmented = cleanup._augment_growth_veto_candidate_shifted_iou(  # pylint: disable=protected-access
        rows,
        sessions,
        gate=cleanup.GrowthVetoGate(max_registered_iou=0.60),
        n_sessions=2,
    )

    assert augmented[0]["shifted_iou"] != augmented[0]["shifted_iou"]


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


def _fake_session(roi_indices: list[int], offset: int) -> object:
    masks = np.zeros((len(roi_indices), 4, 4), dtype=bool)
    for index in range(len(roi_indices)):
        masks[index, offset : offset + 2, index : index + 2] = True
    return SimpleNamespace(
        roi_indices=np.asarray(roi_indices, dtype=int),
        plane_data=SimpleNamespace(
            roi_indices=np.asarray(roi_indices, dtype=int),
            n_rois=len(roi_indices),
            roi_masks=masks,
        ),
    )
