from __future__ import annotations

import numpy as np
from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_teacher_veto_cleanup as veto
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    ResidualFeature,
)


def _weak_feature() -> ResidualFeature:
    return ResidualFeature(
        registered_iou=0.20,
        centroid_distance=4.0,
        area_ratio=0.60,
        row_rank=1,
        column_rank=1,
        row_margin=0.05,
        column_margin=0.10,
        threshold=0.16,
        threshold_margin=0.04,
        assigned_by_hungarian=1,
    )


def test_teacher_veto_cleanup_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-teacher-veto-cleanup"]

    assert canonical == "track2p-policy-teacher-veto-cleanup"
    assert cli._BENCHMARK_ALIASES["track2p-component-teacher-veto-cleanup"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_teacher_veto_cleanup"
    )


def test_teacher_veto_splits_weak_teacher_absent_edge() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
    )

    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_split_edge"
    np.testing.assert_array_equal(report.tracks, [[10, 11, -1, -1], [-1, -1, 12, -1]])


def test_teacher_veto_keeps_teacher_supported_edge() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, 12, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
    )

    assert report.rows == ()
    np.testing.assert_array_equal(report.tracks, predicted)


def test_teacher_veto_rejects_high_margin_edge() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)
    strong = ResidualFeature(
        registered_iou=0.60,
        centroid_distance=1.0,
        area_ratio=0.90,
        row_rank=1,
        column_rank=1,
        row_margin=0.30,
        column_margin=0.40,
        threshold=0.16,
        threshold_margin=0.30,
        assigned_by_hungarian=1,
    )

    report = veto.apply_teacher_veto_edges(
        predicted, teacher, feature_index={edge: strong}
    )

    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "threshold_margin_too_high"
    np.testing.assert_array_equal(report.tracks, predicted)


def test_teacher_veto_rejects_complete_track_by_default() -> None:
    predicted = np.asarray([[10, 11, 12]], dtype=int)
    teacher = np.asarray([[10, 11, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
    )

    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "would_split_complete_track"
    np.testing.assert_array_equal(report.tracks, predicted)
