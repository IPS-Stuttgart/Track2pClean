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


@pytest.fixture()
def calibrated_mht_module(monkeypatch: pytest.MonkeyPatch):
    tracking = types.ModuleType("pyrecest.tracking")
    tracking.ResidualEditCandidate = _ResidualEditCandidate
    tracking.ResidualMHTConfig = _ResidualMHTConfig
    tracking.enumerate_residual_hypotheses = lambda *args, **kwargs: ()
    tracking.select_residual_hypothesis = lambda *args, **kwargs: None

    pyrecest = types.ModuleType("pyrecest")
    pyrecest.tracking = tracking

    monkeypatch.setitem(sys.modules, "pyrecest", pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.tracking", tracking)
    residual_module = (
        "bayescatrack.experiments.track2p_policy_pyrecest_residual_mht_cleanup"
    )
    module_name = (
        "bayescatrack.experiments.track2p_policy_pyrecest_calibrated_mht_cleanup"
    )
    sys.modules.pop(residual_module, None)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)
    yield module

    sys.modules.pop(module_name, None)
    sys.modules.pop(residual_module, None)


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


def _hypothesis(*candidate_ids: str) -> types.SimpleNamespace:
    """Build a minimal stand-in for a PyRecEst ``ResidualHypothesis``."""

    return types.SimpleNamespace(
        candidate_ids=tuple(candidate_ids),
        n_edits=len(candidate_ids),
        score=0.0,
    )


def _split_candidate(module, **overrides: object) -> dict[str, object]:
    """Build a growth-veto split candidate row with a PyRecEst candidate id."""

    row = _candidate_row(**overrides)
    row["pyrecest_candidate_id"] = module._candidate_id(row)  # pylint: disable=protected-access
    return row


def test_short_fragment_count_counts_singletons(residual_mht_module) -> None:
    import numpy as np

    matrix = np.array(
        [
            [10, 11, 12, 13, 14],  # complete track, not a fragment
            [20, 21, -1, -1, -1],  # length two, not short at default threshold
            [-1, -1, -1, -1, 30],  # singleton fragment
        ]
    )

    count = residual_mht_module._track_short_fragment_count(  # pylint: disable=protected-access
        matrix,
        min_meaningful_track_length=2,
    )

    assert count == 1


def test_global_rescore_prefers_less_fragmenting_hypothesis(
    residual_mht_module,
) -> None:
    import numpy as np

    base_tracks = np.array([[10, 11, 12, 13, 14]])
    terminal = _split_candidate(
        residual_mht_module,
        subject="jmX",
        session_a=3,
        session_b=4,
        roi_a=13,
        roi_b=14,
        growth_residual_mahalanobis=5.0,
        growth_residual=0.0,
        registered_iou=0.0,
        is_terminal_edge=1,
        is_last_session_edge=1,
        complete_component_size=5,
    )
    middle = _split_candidate(
        residual_mht_module,
        subject="jmX",
        session_a=2,
        session_b=3,
        roi_a=12,
        roi_b=13,
        growth_residual_mahalanobis=80.0,
        growth_residual=0.0,
        registered_iou=0.0,
        is_terminal_edge=0,
        is_last_session_edge=0,
        complete_component_size=5,
    )
    terminal_id = str(terminal["pyrecest_candidate_id"])
    middle_id = str(middle["pyrecest_candidate_id"])
    options = residual_mht_module.PyRecEstResidualMHTOptions(
        selection_mode="global-rescore",
        edit_penalty=0.25,
        fragmentation_penalty=0.5,
        score_threshold=1.0,
    )
    gate = cleanup.GrowthVetoGate(max_local_neighbor_distortion=None)

    # The additive ranking that PyRecEst would use prefers the two-edit set,
    # because it sums the high-residual middle edit and the terminal edit.
    additive_score = {
        terminal_id: residual_mht_module._candidate_score(terminal, options=options)  # pylint: disable=protected-access
        - options.edit_penalty,
        middle_id: residual_mht_module._candidate_score(middle, options=options)  # pylint: disable=protected-access
        - options.edit_penalty,
        f"{terminal_id};{middle_id}": (
            residual_mht_module._candidate_score(terminal, options=options)  # pylint: disable=protected-access
            + residual_mht_module._candidate_score(middle, options=options)  # pylint: disable=protected-access
            - 2 * options.edit_penalty
        ),
    }
    assert additive_score[f"{terminal_id};{middle_id}"] > additive_score[middle_id]

    hypotheses = [
        _hypothesis(),
        _hypothesis(terminal_id),
        _hypothesis(middle_id),
        _hypothesis(terminal_id, middle_id),
    ]

    selected, objective = (
        residual_mht_module._select_residual_hypothesis_global_rescore(  # pylint: disable=protected-access
            hypotheses,
            [terminal, middle],
            base_tracks=base_tracks,
            gate=gate,
            options=options,
        )
    )

    # Global re-scoring rejects the two-edit set because applying both shatters
    # the single complete track into two singletons, and keeps the clean,
    # high-residual middle split alone.
    assert selected.candidate_ids == (middle_id,)
    assert objective == pytest.approx(2.95)


def test_global_rescore_falls_back_to_no_edit_below_threshold(
    residual_mht_module,
) -> None:
    import numpy as np

    base_tracks = np.array([[10, 11, 12, 13, 14]])
    middle = _split_candidate(
        residual_mht_module,
        subject="jmX",
        session_a=2,
        session_b=3,
        roi_a=12,
        roi_b=13,
        growth_residual_mahalanobis=80.0,
        complete_component_size=5,
    )
    middle_id = str(middle["pyrecest_candidate_id"])
    options = residual_mht_module.PyRecEstResidualMHTOptions(
        selection_mode="global-rescore",
        score_threshold=5.0,
    )
    gate = cleanup.GrowthVetoGate(max_local_neighbor_distortion=None)
    empty = _hypothesis()

    selected, objective = (
        residual_mht_module._select_residual_hypothesis_global_rescore(  # pylint: disable=protected-access
            [empty, _hypothesis(middle_id)],
            [middle],
            base_tracks=base_tracks,
            gate=gate,
            options=options,
        )
    )

    assert selected is empty
    assert selected.candidate_ids == ()
    assert objective == 0.0


def test_global_rescore_does_not_read_audit_only_columns(
    residual_mht_module,
) -> None:
    import numpy as np

    base_tracks = np.array([[10, 11, 12, 13, 14]])
    middle = _AuditGuardRow(
        _split_candidate(
            residual_mht_module,
            subject="jmX",
            session_a=2,
            session_b=3,
            roi_a=12,
            roi_b=13,
            growth_residual_mahalanobis=80.0,
            complete_component_size=5,
            edge_status_against_gt="false_positive",
            complete_fp_delta_if_removed=-1,
            score_delta=100.0,
        )
    )
    middle_id = str(dict.__getitem__(middle, "pyrecest_candidate_id"))
    options = residual_mht_module.PyRecEstResidualMHTOptions(
        selection_mode="global-rescore",
    )
    gate = cleanup.GrowthVetoGate(max_local_neighbor_distortion=None)

    selected, _objective = (
        residual_mht_module._select_residual_hypothesis_global_rescore(  # pylint: disable=protected-access
            [_hypothesis(), _hypothesis(middle_id)],
            [middle],
            base_tracks=base_tracks,
            gate=gate,
            options=options,
        )
    )

    assert selected.candidate_ids == (middle_id,)


def test_residual_mht_cli_exposes_global_rescore_knobs(residual_mht_module) -> None:
    args = residual_mht_module.build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "mht.csv",
            "--mht-selection-mode",
            "global-rescore",
            "--mht-fragmentation-penalty",
            "0.75",
        ]
    )

    assert args.mht_selection_mode == "global-rescore"
    assert args.mht_fragmentation_penalty == 0.75


def test_calibrated_mht_cli_exposes_fold_calibration_knobs(
    calibrated_mht_module,
) -> None:
    args = calibrated_mht_module.build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "mht.csv",
            "--calibrated-fp-logistic-c",
            "0.25",
            "--mht-score-threshold",
            "0",
        ]
    )

    assert args.calibrated_fp_logistic_c == 0.25
    assert args.mht_score_threshold == 0.0


def test_calibrated_threshold_is_single_training_fold_decision(
    calibrated_mht_module,
) -> None:
    false_positive = _candidate_row(
        edge_status_against_gt="false_positive",
        pairwise_fp_delta_if_removed=-1,
        pairwise_tp_delta_if_removed=0,
        pairwise_fn_delta_if_removed=0,
        complete_fp_delta_if_removed=-1,
        complete_tp_delta_if_removed=0,
        complete_fn_delta_if_removed=0,
    )
    unsafe_true_positive = _candidate_row(
        roi_b=100,
        edge_status_against_gt="true_positive",
        pairwise_fp_delta_if_removed=0,
        pairwise_tp_delta_if_removed=-1,
        pairwise_fn_delta_if_removed=1,
        complete_fp_delta_if_removed=0,
        complete_tp_delta_if_removed=-1,
        complete_fn_delta_if_removed=1,
    )

    threshold = calibrated_mht_module._select_training_probability_threshold(  # pylint: disable=protected-access
        [false_positive, unsafe_true_positive],
        [0.8, 0.7],
    )

    assert threshold == pytest.approx(0.8)


def test_calibrated_heldout_scoring_does_not_read_audit_only_columns(
    calibrated_mht_module,
) -> None:
    row = _AuditGuardRow(
        _candidate_row(
            edge_status_against_gt="false_positive",
            pairwise_fp_delta_if_removed=-1,
            pairwise_tp_delta_if_removed=0,
            complete_fp_delta_if_removed=-1,
            complete_tp_delta_if_removed=0,
            manual_gt_reference_identity="jm038:true-track",
            new_complete_track_f1_micro=0.974,
            score_delta=100.0,
        )
    )
    calibrator = calibrated_mht_module._constant_false_positive_calibrator(  # pylint: disable=protected-access
        0.8
    )

    candidates = calibrated_mht_module._calibrated_candidate_rows(  # pylint: disable=protected-access
        [row],
        gate=cleanup.GrowthVetoGate(max_local_neighbor_distortion=None),
        n_sessions=7,
        calibrator=calibrator,
        threshold=0.5,
    )
    pyrecest_candidate = (
        calibrated_mht_module._to_calibrated_pyrecest_candidate(  # pylint: disable=protected-access
            candidates[0],
        )
    )

    assert len(candidates) == 1
    assert candidates[0]["pyrecest_candidate_family"] == "calibrated_fp_probability"
    assert candidates[0]["calibrated_fp_probability"] == pytest.approx(0.8)
    assert pyrecest_candidate.score > 0.0
    assert "calibrated_fp_probability" in pyrecest_candidate.metadata


def test_calibrated_score_is_threshold_relative_log_likelihood(
    calibrated_mht_module,
) -> None:
    score_at_threshold = calibrated_mht_module._calibrated_log_likelihood_score(  # pylint: disable=protected-access
        0.75,
        0.75,
    )
    score_above_threshold = calibrated_mht_module._calibrated_log_likelihood_score(  # pylint: disable=protected-access
        0.9,
        0.75,
    )

    assert score_at_threshold == pytest.approx(0.0)
    assert score_above_threshold > 0.0
