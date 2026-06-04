from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

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


FORBIDDEN_AUDIT_COLUMNS = frozenset(
    {
        "edge_status_against_gt",
        "pairwise_tp_delta_if_removed",
        "pairwise_fp_delta_if_removed",
        "pairwise_fn_delta_if_removed",
        "complete_tp_delta_if_removed",
        "complete_fp_delta_if_removed",
        "complete_fn_delta_if_removed",
        "reference_track_id",
        "reference_track_identity",
        "reference_identity",
        "manual_gt_reference_identity",
        "manual_gt_track_id",
        "nearest_gt_track_id",
        "new_pairwise_f1_micro",
        "new_complete_track_f1_micro",
        "score_delta",
    }
)


def _is_audit_only_column(key: str) -> bool:
    return (
        key in FORBIDDEN_AUDIT_COLUMNS
        or "against_gt" in key
        or key.endswith("_delta_if_removed")
        or key.startswith("manual_gt_")
        or key.startswith("new_pairwise_")
        or key.startswith("new_complete_")
    )


class _AuditGuardRow(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if _is_audit_only_column(key):
            raise AssertionError(f"selector read audit-only column {key!r}")
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if _is_audit_only_column(key):
            raise AssertionError(f"selector read audit-only column {key!r}")
        return super().get(key, default)


def _edge_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(row["session_a"]),
        int(row["session_b"]),
        int(row["roi_a"]),
        int(row["roi_b"]),
        int(row.get("occurrence_index", 0)),
    )


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
    assert args.max_veto_registered_iou == 0.60
    assert args.max_veto_shifted_iou == 0.80
    assert args.min_veto_cell_probability == 0.50
    assert args.max_veto_min_cell_probability == 0.65
    assert args.max_veto_row_rank == 1
    assert args.max_veto_column_rank == 1
    assert args.require_veto_not_suffix_edge is True
    assert args.max_vetoes_per_subject == 1


def test_growth_veto_gate_accepts_extreme_terminal_complete_edge() -> None:
    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(), cleanup.GrowthVetoGate(), n_sessions=7
    )

    assert reason == "accepted"


def test_growth_veto_gate_accepts_weak_local_evidence_pocket() -> None:
    gate = cleanup.GrowthVetoGate(
        min_growth_residual_mahalanobis=10.0,
        min_registered_iou=0.0,
        max_registered_iou=0.60,
        min_shifted_iou=0.0,
        min_cell_probability=0.0,
        max_min_cell_probability=0.60,
        min_anchor_count=2,
    )

    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(
            growth_residual_mahalanobis=26.07,
            registered_iou=0.552,
            shifted_iou=0.50,
            cell_probability_a=0.571,
            cell_probability_b=0.806,
        ),
        gate,
        n_sessions=7,
    )

    assert reason == "accepted"


def test_growth_veto_gate_can_reject_high_registered_iou_true_edge_tail() -> None:
    gate = cleanup.GrowthVetoGate(
        min_growth_residual_mahalanobis=10.0,
        min_registered_iou=0.0,
        max_registered_iou=0.60,
        min_shifted_iou=0.0,
        min_cell_probability=0.0,
        max_min_cell_probability=0.60,
    )

    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(registered_iou=0.688, cell_probability_a=0.571),
        gate,
        n_sessions=7,
    )

    assert reason == "registered_iou_above_gate"


def test_growth_veto_gate_can_reject_high_cell_confidence_true_edge_tail() -> None:
    gate = cleanup.GrowthVetoGate(
        max_min_cell_probability=0.60,
    )

    reason = cleanup.growth_veto_gate_reason(
        _candidate_row(cell_probability_a=0.748, cell_probability_b=0.912),
        gate,
        n_sessions=7,
    )

    assert reason == "min_cell_probability_above_gate"


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


def test_growth_veto_sparse_shifted_iou_prefilter_respects_upper_bounds(
    monkeypatch,
) -> None:
    rows = [
        _candidate_row(session_a=0, session_b=1, shifted_iou=float("nan"), roi_b=1210),
        _candidate_row(
            session_a=0,
            session_b=1,
            shifted_iou=float("nan"),
            roi_b=1211,
            registered_iou=0.80,
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
            "shifted_iou": np.asarray([[0.76]], dtype=float)
        },
    )

    augmented = cleanup._augment_growth_veto_candidate_shifted_iou(  # pylint: disable=protected-access
        rows,
        sessions,
        gate=cleanup.GrowthVetoGate(max_registered_iou=0.60),
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


def test_growth_veto_selector_ignores_audit_only_gt_and_delta_columns() -> None:
    rows = [
        _candidate_row(
            growth_residual_mahalanobis=26.0,
            roi_b=1210,
            edge_status_against_gt="false_positive",
            pairwise_fp_delta_if_removed=-1,
            complete_fp_delta_if_removed=-1,
            complete_tp_delta_if_removed=0,
            reference_track_id="jm046:manual-fp",
            new_pairwise_f1_micro=0.967213,
            new_complete_track_f1_micro=0.957983,
        ),
        _candidate_row(
            growth_residual_mahalanobis=30.0,
            roi_b=1211,
            edge_status_against_gt="true_positive",
            pairwise_fp_delta_if_removed=0,
            complete_fp_delta_if_removed=0,
            complete_tp_delta_if_removed=-1,
            manual_gt_reference_identity="jm046:true-track",
            score_delta=-100.0,
            new_pairwise_f1_micro=0.1,
            new_complete_track_f1_micro=0.1,
        ),
    ]
    sanitized_rows = [
        {key: value for key, value in row.items() if not _is_audit_only_column(key)}
        for row in rows
    ]

    selected = cleanup._selected_growth_veto_rows(  # pylint: disable=protected-access
        rows, gate=cleanup.GrowthVetoGate(max_vetoes_per_subject=1), n_sessions=7
    )
    sanitized_selected = (
        cleanup._selected_growth_veto_rows(  # pylint: disable=protected-access
            sanitized_rows,
            gate=cleanup.GrowthVetoGate(max_vetoes_per_subject=1),
            n_sessions=7,
        )
    )

    assert [_edge_key(row) for row in selected] == [
        _edge_key(row) for row in sanitized_selected
    ]


def test_growth_veto_selector_does_not_access_audit_only_columns() -> None:
    rows = [
        _AuditGuardRow(
            _candidate_row(
                growth_residual_mahalanobis=30.0,
                roi_b=1211,
                edge_status_against_gt="true_positive",
                pairwise_fp_delta_if_removed=0,
                complete_fp_delta_if_removed=0,
                complete_tp_delta_if_removed=-1,
                reference_track_id="manual-track",
                score_delta=-1.0,
            )
        ),
        _AuditGuardRow(
            _candidate_row(
                growth_residual_mahalanobis=26.0,
                roi_b=1210,
                edge_status_against_gt="false_positive",
                pairwise_fp_delta_if_removed=-1,
                complete_fp_delta_if_removed=-1,
                complete_tp_delta_if_removed=0,
                reference_track_id="manual-fp",
                score_delta=1.0,
            )
        ),
    ]

    selected = cleanup._selected_growth_veto_rows(  # pylint: disable=protected-access
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
