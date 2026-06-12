from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup


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


@dataclass(frozen=True)
class _ResidualEditCandidate:
    candidate_id: str
    score: float
    conflict_keys: frozenset[str]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _ResidualMHTConfig:
    max_edits: int
    max_hypotheses: int
    edit_penalty: float
    score_threshold: float
    include_empty: bool


def _candidate_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "subject": "jm038",
        "session_a": 5,
        "session_b": 6,
        "roi_a": 32,
        "roi_b": 99,
        "occurrence_index": 0,
        "remove_reason": "split_edge",
        "edge_source": "policy",
        "would_split_component": 1,
        "is_terminal_edge": 1,
        "is_last_session_edge": 1,
        "complete_component_size": 7,
        "growth_anchor_count": 2,
        "growth_residual_mahalanobis": 1.37,
        "growth_residual": 0.46,
        "registered_iou": 0.88,
        "shifted_iou": float("nan"),
        "row_rank": 1,
        "column_rank": 1,
        "cell_probability_a": 0.89,
        "cell_probability_b": 0.91,
        "local_neighbor_distortion": 0.0,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def residual_mht_module(monkeypatch: pytest.MonkeyPatch):
    tracking = types.ModuleType("pyrecest.tracking")
    tracking.ResidualEditCandidate = _ResidualEditCandidate
    tracking.ResidualMHTConfig = _ResidualMHTConfig
    tracking.enumerate_residual_hypotheses = lambda *args, **kwargs: ()
    tracking.select_residual_hypothesis = lambda *args, **kwargs: None

    pyrecest = types.ModuleType("pyrecest")
    pyrecest.tracking = tracking

    monkeypatch.setitem(sys.modules, "pyrecest", pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.tracking", tracking)
    module_name = (
        "bayescatrack.experiments.track2p_policy_pyrecest_residual_mht_cleanup"
    )
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)
    yield module

    sys.modules.pop(module_name, None)


def test_high_overlap_pocket_is_opt_in(residual_mht_module) -> None:
    options = residual_mht_module.PyRecEstResidualMHTOptions(
        include_high_overlap_low_motion=False,
    )

    reason = residual_mht_module._high_overlap_low_motion_reason(  # pylint: disable=protected-access
        _candidate_row(),
        gate=cleanup.GrowthVetoGate(max_local_neighbor_distortion=None),
        options=options,
        n_sessions=7,
    )

    assert reason == "high_overlap_low_motion_disabled"


def test_high_overlap_pocket_exposes_cell_probability_gate(
    residual_mht_module,
) -> None:
    args = residual_mht_module.build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "mht.csv",
            "--mht-high-overlap-min-cell-probability",
            "0.75",
        ]
    )

    assert args.mht_high_overlap_min_cell_probability == 0.75


def test_high_overlap_pocket_can_require_stronger_endpoint_cell_probability(
    residual_mht_module,
) -> None:
    options = residual_mht_module.PyRecEstResidualMHTOptions(
        include_high_overlap_low_motion=True,
        high_overlap_min_cell_probability=0.75,
    )
    gate = cleanup.GrowthVetoGate(
        min_cell_probability=0.50,
        max_local_neighbor_distortion=None,
    )

    reason = residual_mht_module._high_overlap_low_motion_reason(  # pylint: disable=protected-access
        _candidate_row(cell_probability_a=0.74, cell_probability_b=0.91),
        gate=gate,
        options=options,
        n_sessions=7,
    )

    assert reason == "high_overlap_cell_probability_below_gate"


def test_high_overlap_pocket_does_not_read_audit_only_columns(
    residual_mht_module,
) -> None:
    row = _AuditGuardRow(
        _candidate_row(
            edge_status_against_gt="false_positive",
            pairwise_fp_delta_if_removed=-1,
            pairwise_tp_delta_if_removed=0,
            complete_fp_delta_if_removed=-1,
            complete_tp_delta_if_removed=0,
            reference_track_id="jm038:manual-fp",
            manual_gt_reference_identity="jm038:true-track",
            new_pairwise_f1_micro=0.966311,
            new_complete_track_f1_micro=0.966102,
            score_delta=100.0,
        )
    )
    options = residual_mht_module.PyRecEstResidualMHTOptions(
        include_high_overlap_low_motion=True,
        high_overlap_min_registered_iou=0.85,
        high_overlap_max_growth_residual=0.50,
        high_overlap_min_growth_residual_mahalanobis=1.0,
        high_overlap_min_cell_probability=0.75,
        high_overlap_score_bonus=2.0,
    )
    gate = cleanup.GrowthVetoGate(
        min_growth_residual_mahalanobis=20.0,
        min_growth_residual=2.5,
        min_registered_iou=0.45,
        max_registered_iou=0.60,
        min_shifted_iou=0.60,
        max_shifted_iou=0.80,
        min_cell_probability=0.50,
        max_min_cell_probability=0.65,
        max_local_neighbor_distortion=None,
        max_row_rank=1,
        max_column_rank=1,
        require_not_suffix_edge=True,
        require_terminal_edge=True,
        require_last_session_edge=True,
        require_complete_component=True,
    )

    candidates = residual_mht_module._candidate_rows(  # pylint: disable=protected-access
        [row],
        gate=gate,
        options=options,
        n_sessions=7,
    )
    pyrecest_candidate = residual_mht_module._to_pyrecest_candidate(  # pylint: disable=protected-access
        candidates[0],
        options=options,
    )

    assert len(candidates) == 1
    assert candidates[0]["pyrecest_candidate_family"] == "high_overlap_low_motion"
    assert pyrecest_candidate.candidate_id == "jm038:5:6:32:99:0"
    assert pyrecest_candidate.score > 0.0
    assert set(pyrecest_candidate.metadata) == {
        "subject",
        "session_a",
        "session_b",
        "roi_a",
        "roi_b",
    }


def test_high_overlap_selection_matches_sanitized_rows(residual_mht_module) -> None:
    rows = [
        _candidate_row(
            roi_b=99,
            registered_iou=0.88,
            edge_status_against_gt="false_positive",
            complete_fp_delta_if_removed=-1,
            score_delta=1000.0,
        ),
        _candidate_row(
            roi_b=100,
            registered_iou=0.86,
            edge_status_against_gt="true_positive",
            complete_tp_delta_if_removed=-1,
            manual_gt_reference_identity="jm038:true-track",
            score_delta=-1000.0,
        ),
    ]
    sanitized_rows = [
        {key: value for key, value in row.items() if not _is_audit_only_column(key)}
        for row in rows
    ]
    options = residual_mht_module.PyRecEstResidualMHTOptions(
        candidate_top_k=2,
        include_high_overlap_low_motion=True,
        high_overlap_min_registered_iou=0.85,
        high_overlap_max_growth_residual=0.50,
        high_overlap_min_growth_residual_mahalanobis=1.0,
    )
    gate = cleanup.GrowthVetoGate(
        min_growth_residual_mahalanobis=20.0,
        min_growth_residual=2.5,
        max_registered_iou=0.60,
        max_shifted_iou=0.80,
        max_local_neighbor_distortion=None,
        require_not_suffix_edge=True,
        require_terminal_edge=True,
        require_last_session_edge=True,
        require_complete_component=True,
    )

    selected = residual_mht_module._candidate_rows(  # pylint: disable=protected-access
        rows,
        gate=gate,
        options=options,
        n_sessions=7,
    )
    sanitized_selected = residual_mht_module._candidate_rows(  # pylint: disable=protected-access
        sanitized_rows,
        gate=gate,
        options=options,
        n_sessions=7,
    )

    assert [row["pyrecest_candidate_id"] for row in selected] == [
        row["pyrecest_candidate_id"] for row in sanitized_selected
    ]
