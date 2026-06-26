from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bayescatrack.experiments import full_mht_scan_history_dynamics_integration as scan_dynamics
from bayescatrack.experiments.full_mht_scan_history_dynamics_integration import (
    ScanHistoryEdgeFeatures,
    parse_selected_edge_summary,
    row_scan_motion_history_risk,
    scan_motion_history_risk,
)


def _feature(
    edge: tuple[int, int, int, int] = (0, 1, 1, 20),
    *,
    registered_iou: float = 0.80,
    shifted_iou: float = 0.70,
    growth_residual: float = 1.0,
    growth_mahalanobis: float = 1.0,
    local_deformation: float = 0.10,
) -> ScanHistoryEdgeFeatures:
    return ScanHistoryEdgeFeatures(
        edge=edge,
        registered_iou=registered_iou,
        shifted_iou=shifted_iou,
        growth_residual=growth_residual,
        growth_mahalanobis=growth_mahalanobis,
        local_deformation=local_deformation,
    )


def _summary(edge: tuple[int, int, int, int], **overrides: float) -> str:
    values = {
        "reg": 0.80,
        "shift": 0.70,
        "growth": 1.0,
        "mahal": 1.0,
        "local": 0.10,
    }
    values.update(overrides)
    session_a, session_b, roi_a, roi_b = edge
    return (
        f"{session_a}:{roi_a}->{session_b}:{roi_b}"
        "|prior=1|score=1|risk=0|veto=prior_veto_disabled"
        f"|reg={values['reg']}|shift={values['shift']}"
        f"|growth={values['growth']}|mahal={values['mahal']}"
        f"|local={values['local']}|cell=0.9|row_rank=1|column_rank=1"
    )


def test_parse_selected_edge_summary_reads_label_free_features() -> None:
    parsed = parse_selected_edge_summary(
        "5:2309->6:1210|prior=1|reg=0.5|shift=0.7|growth=2.5|mahal=20|local=0.2"
    )

    assert parsed is not None
    assert parsed.edge == (5, 6, 2309, 1210)
    assert parsed.registered_iou == pytest.approx(0.5)
    assert parsed.growth_mahalanobis == pytest.approx(20.0)


def test_row_scan_motion_history_risk_ignores_coherent_edges() -> None:
    risk = row_scan_motion_history_risk(
        [
            _feature(edge=(0, 1, 1, 20), registered_iou=0.80, growth_residual=1.0),
            _feature(edge=(1, 2, 20, 30), registered_iou=0.79, growth_residual=1.1),
            _feature(edge=(2, 3, 30, 40), registered_iou=0.82, growth_residual=0.9),
        ]
    )

    assert risk == pytest.approx(0.0)


def test_row_scan_motion_history_risk_penalizes_outlier_edges() -> None:
    risk = row_scan_motion_history_risk(
        [
            _feature(edge=(0, 1, 1, 20)),
            _feature(edge=(1, 2, 20, 30)),
            _feature(
                edge=(2, 3, 30, 41),
                registered_iou=0.30,
                shifted_iou=0.20,
                growth_residual=8.0,
                growth_mahalanobis=8.0,
                local_deformation=1.20,
            ),
        ]
    )

    assert risk > 6.0


def test_scan_motion_history_risk_groups_edges_by_identity_row() -> None:
    hypothesis = SimpleNamespace(
        tracks=np.asarray([[1, 20, 30], [2, 21, 31]], dtype=int),
        history=(
            {
                "selected_edge_summaries": ";".join(
                    [
                        _summary((0, 1, 1, 20)),
                        _summary((0, 1, 2, 21)),
                    ]
                )
            },
            {
                "selected_edge_summaries": ";".join(
                    [
                        _summary((1, 2, 20, 30), reg=0.20, shift=0.20, growth=9.0, mahal=9.0, local=1.3),
                        _summary((1, 2, 21, 31)),
                    ]
                )
            },
        ),
    )

    assert scan_motion_history_risk(hypothesis) > 6.0


def test_scan_history_pruning_can_keep_lower_score_coherent_history() -> None:
    pytest.importorskip("pyrecest")
    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    scan_dynamics.install_full_mht_scan_history_dynamics_pruning()

    bad_local_history = full_mht._MHTHypothesis(
        np.asarray([[1, 20, 41]], dtype=int),
        10.0,
        (
            {"selected_edge_summaries": _summary((0, 1, 1, 20))},
            {
                "selected_edge_summaries": _summary(
                    (1, 2, 20, 41),
                    reg=0.20,
                    shift=0.20,
                    growth=9.0,
                    mahal=9.0,
                    local=1.3,
                )
            },
        ),
    )
    lower_score_coherent_history = full_mht._MHTHypothesis(
        np.asarray([[1, 20, 30]], dtype=int),
        9.0,
        (
            {"selected_edge_summaries": _summary((0, 1, 1, 20))},
            {"selected_edge_summaries": _summary((1, 2, 20, 30))},
        ),
    )
    config = full_mht.FullMHTConfig(beam_width=1)
    object.__setattr__(config, "scan_motion_history_weight", 1.0)

    selected = full_mht._prune_beam(
        [bad_local_history, lower_score_coherent_history],
        config=config,
    )

    assert selected == [lower_score_coherent_history]
