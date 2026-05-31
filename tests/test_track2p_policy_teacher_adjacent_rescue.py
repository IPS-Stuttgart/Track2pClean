from __future__ import annotations

import numpy as np
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    ResidualFeature,
)
from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
    TeacherEdgeFeatureGate,
    apply_teacher_adjacent_rescue_edges,
)


def test_teacher_adjacent_rescue_inserts_missing_source_into_seed_anchored_row() -> (
    None
):
    predicted = np.asarray([[100, -1, -1, 40]], dtype=int)
    teacher = np.asarray([[100, -1, 30, 40]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
    )

    np.testing.assert_array_equal(output.tracks, [[100, -1, 30, 40]])
    assert output.rows == (
        {
            "session_a": 2,
            "session_b": 3,
            "roi_a": 30,
            "roi_b": 40,
            "applied": 1,
            "reason": "accepted_insert_source",
            "source_row": -1,
            "target_row": 0,
            "teacher_complete_row_supported": 0,
            "occurrence_index": 0,
        },
    )


def test_teacher_adjacent_rescue_can_disable_source_insertions_alias() -> None:
    predicted = np.asarray([[100, -1, -1, 40]], dtype=int)
    teacher = np.asarray([[100, -1, 30, 40]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_source_insertions=False,
    )

    np.testing.assert_array_equal(output.tracks, predicted)
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "missing_or_ambiguous_source"


def test_teacher_adjacent_rescue_can_require_component_support() -> None:
    predicted = np.asarray([[100, -1, -1, -1]], dtype=int)
    teacher = np.asarray([[100, 20, -1, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        min_component_observations=2,
    )

    np.testing.assert_array_equal(output.tracks, predicted)
    assert output.rows == (
        {
            "session_a": 0,
            "session_b": 1,
            "roi_a": 100,
            "roi_b": 20,
            "applied": 0,
            "reason": "insufficient_component_support",
            "source_row": 0,
            "target_row": -1,
            "teacher_complete_row_supported": 0,
            "occurrence_index": 0,
        },
    )


def test_teacher_adjacent_rescue_rejects_source_insertion_that_completes_row() -> None:
    predicted = np.asarray([[100, 20, -1, 40]], dtype=int)
    teacher = np.asarray([[-1, -1, 30, 40]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
    )

    np.testing.assert_array_equal(output.tracks, predicted)
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "would_complete_track"


def test_teacher_adjacent_rescue_rejects_seed_source_completion_by_default() -> None:
    predicted = np.asarray([[-1, 20, 30, 40]], dtype=int)
    teacher = np.asarray([[100, 20, -1, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
    )

    np.testing.assert_array_equal(output.tracks, predicted)
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "would_complete_track"


def test_teacher_adjacent_rescue_can_complete_seed_source_backfill() -> None:
    predicted = np.asarray([[-1, 20, 30, 40]], dtype=int)
    teacher = np.asarray([[100, 20, -1, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
    )

    np.testing.assert_array_equal(output.tracks, [[100, 20, 30, 40]])
    assert output.rows[0]["applied"] == 1
    assert output.rows[0]["reason"] == "accepted_insert_source"


def test_teacher_adjacent_rescue_accepts_seed_completion_alias() -> None:
    predicted = np.asarray([[-1, 20, 30, 40]], dtype=int)
    teacher = np.asarray([[100, 20, -1, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        allow_seed_completing_rescue=True,
    )

    np.testing.assert_array_equal(output.tracks, [[100, 20, 30, 40]])
    assert output.rows[0]["applied"] == 1
    assert output.rows[0]["reason"] == "accepted_insert_source"


def test_teacher_adjacent_rescue_accepts_seed_completing_backfill_alias() -> None:
    predicted = np.asarray([[-1, 11, 12]], dtype=int)
    teacher = np.asarray([[10, 11, -1]], dtype=int)

    default_report = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
    )
    np.testing.assert_array_equal(default_report.tracks, predicted)
    assert default_report.rows[0]["applied"] == 0
    assert default_report.rows[0]["reason"] == "would_complete_track"

    opt_in_report = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        allow_seed_completing_backfill=True,
    )
    np.testing.assert_array_equal(opt_in_report.tracks, [[10, 11, 12]])
    assert opt_in_report.rows[0]["applied"] == 1
    assert opt_in_report.rows[0]["reason"] == "accepted_insert_source"


def test_seed_completing_backfill_alias_does_not_allow_nonseed_completion() -> None:
    predicted = np.asarray([[10, 11, -1, 13]], dtype=int)
    teacher = np.asarray([[-1, -1, 12, 13]], dtype=int)

    report = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_completing_backfill=True,
    )

    np.testing.assert_array_equal(report.tracks, predicted)
    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "would_complete_track"


def test_teacher_adjacent_rescue_allows_seed_anchored_fragment_merge_from_target() -> (
    None
):
    predicted = np.asarray([[100, -1, 30, -1], [-1, -1, -1, 40]], dtype=int)
    teacher = np.asarray([[-1, -1, 30, 40]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
    )

    np.testing.assert_array_equal(output.tracks, [[100, -1, 30, 40]])
    assert output.rows[0]["applied"] == 1
    assert output.rows[0]["reason"] == "accepted_merge_fragments"


def test_teacher_adjacent_rescue_teacher_complete_row_alias_reports_support() -> None:
    predicted = np.asarray([[100, 20, 30, -1]], dtype=int)
    teacher = np.asarray([[100, 20, 30, 40]], dtype=int)

    guarded = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
    )

    np.testing.assert_array_equal(guarded.tracks, predicted)
    assert guarded.rows[-1]["applied"] == 0
    assert guarded.rows[-1]["reason"] == "would_complete_track"
    assert guarded.rows[-1]["teacher_complete_row_supported"] == 1

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_teacher_complete_row_rescue=True,
    )

    np.testing.assert_array_equal(output.tracks, [[100, 20, 30, 40]])
    assert output.rows[-1]["applied"] == 1
    assert output.rows[-1]["reason"] == "accepted_insert_target"
    assert output.rows[-1]["teacher_complete_row_supported"] == 1


def test_teacher_adjacent_rescue_dynamic_confidence_reorders_stale_slot_claims() -> (
    None
):
    """Dynamic confidence should use local evidence before a slot is claimed.

    Two teacher edges compete for the same target ROI. A purely structural order
    would tie-break lexicographically and let the lower-confidence source claim
    the target first. The dynamic-confidence order should instead apply the
    stronger label-free registration edge, leaving the weaker edge rejected by the
    target-source conflict.
    """

    predicted = np.asarray([[100, -1, -1], [200, -1, -1]], dtype=int)
    teacher = np.asarray([[100, 10, -1], [200, 10, -1]], dtype=int)
    edge_feature_index = {
        (0, 1, 100, 10): ResidualFeature(
            registered_iou=0.10,
            centroid_distance=5.0,
            area_ratio=0.50,
            row_margin=0.0,
            column_margin=0.0,
            threshold_margin=0.0,
            assigned_by_hungarian=0,
        ),
        (0, 1, 200, 10): ResidualFeature(
            registered_iou=0.90,
            centroid_distance=1.0,
            area_ratio=0.95,
            row_margin=0.4,
            column_margin=0.4,
            threshold_margin=0.5,
            assigned_by_hungarian=1,
        ),
    }

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        edge_order="dynamic-confidence",
        edge_feature_index=edge_feature_index,
    )

    np.testing.assert_array_equal(output.tracks, [[100, -1, -1], [200, 10, -1]])
    assert output.rows[0]["roi_a"] == 200
    assert output.rows[0]["applied"] == 1
    assert output.rows[0]["reason"] == "accepted_insert_target"
    assert output.rows[1]["roi_a"] == 100
    assert output.rows[1]["applied"] == 0
    assert output.rows[1]["reason"] == "target_has_source_conflict"


def test_teacher_adjacent_rescue_feature_gate_rejects_weak_edge() -> None:
    predicted = np.asarray([[10, -1, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1], [10, 12, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        edge_order="lexicographic",
        edge_feature_index={
            (0, 1, 10, 11): ResidualFeature(
                registered_iou=0.2,
                centroid_distance=1.0,
                area_ratio=0.90,
                threshold_margin=0.30,
                assigned_by_hungarian=1,
            ),
            (0, 1, 10, 12): ResidualFeature(
                registered_iou=0.8,
                centroid_distance=1.0,
                area_ratio=0.95,
                threshold_margin=0.30,
                assigned_by_hungarian=1,
            ),
        },
        feature_gate=TeacherEdgeFeatureGate(min_registered_iou=0.5),
    )

    np.testing.assert_array_equal(output.tracks, [[10, 12, -1]])
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "feature_gate_registered_iou"
    assert output.rows[1]["applied"] == 1
    assert output.rows[1]["reason"] == "accepted_insert_target"


def test_teacher_adjacent_rescue_feature_gate_rejects_missing_features() -> None:
    predicted = np.asarray([[10, -1, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        feature_gate=TeacherEdgeFeatureGate(min_registered_iou=0.5),
    )

    np.testing.assert_array_equal(output.tracks, predicted)
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "feature_gate_missing"


def test_teacher_adjacent_rescue_feature_gate_requires_hungarian() -> None:
    predicted = np.asarray([[10, -1, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1], [10, 12, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        edge_order="lexicographic",
        edge_feature_index={
            (0, 1, 10, 11): ResidualFeature(
                registered_iou=0.8,
                centroid_distance=1.0,
                area_ratio=0.95,
                threshold_margin=0.30,
                assigned_by_hungarian=0,
            ),
            (0, 1, 10, 12): ResidualFeature(
                registered_iou=0.8,
                centroid_distance=1.0,
                area_ratio=0.95,
                threshold_margin=0.30,
                assigned_by_hungarian=1,
            ),
        },
        feature_gate=TeacherEdgeFeatureGate(require_hungarian=True),
    )

    np.testing.assert_array_equal(output.tracks, [[10, 12, -1]])
    assert output.rows[0]["reason"] == "feature_gate_hungarian"
    assert output.rows[1]["reason"] == "accepted_insert_target"
