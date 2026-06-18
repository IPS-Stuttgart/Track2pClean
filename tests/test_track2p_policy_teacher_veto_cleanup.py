from __future__ import annotations

import numpy as np
import pytest
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
        cell_probability_a=0.72,
        cell_probability_b=0.68,
        row_rank=1,
        column_rank=1,
        row_margin=0.05,
        column_margin=0.10,
        threshold=0.16,
        threshold_margin=0.04,
        assigned_by_hungarian=1,
    )


def _weaker_feature() -> ResidualFeature:
    return ResidualFeature(
        registered_iou=0.05,
        centroid_distance=6.0,
        area_ratio=0.50,
        cell_probability_a=0.40,
        cell_probability_b=0.35,
        row_rank=1,
        column_rank=1,
        row_margin=0.01,
        column_margin=0.02,
        threshold=0.16,
        threshold_margin=0.01,
        assigned_by_hungarian=1,
    )


def _feature_with_margin(
    *, threshold_margin: float, row_margin: float, column_margin: float
) -> ResidualFeature:
    return ResidualFeature(
        registered_iou=0.20,
        centroid_distance=4.0,
        area_ratio=0.60,
        cell_probability_a=0.72,
        cell_probability_b=0.68,
        row_rank=1,
        column_rank=1,
        row_margin=row_margin,
        column_margin=column_margin,
        threshold=0.16,
        threshold_margin=threshold_margin,
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
        config=veto.TeacherVetoConfig(min_fragment_observations=1),
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


def test_teacher_veto_rejects_short_fragment_by_default() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
    )

    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "fragment_too_short"
    assert report.rows[0]["left_observations"] == 2
    assert report.rows[0]["right_observations"] == 1
    np.testing.assert_array_equal(report.tracks, predicted)


def test_teacher_veto_rejects_high_margin_edge() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)
    strong = ResidualFeature(
        registered_iou=0.60,
        centroid_distance=1.0,
        area_ratio=0.90,
        cell_probability_a=0.95,
        cell_probability_b=0.93,
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


def test_teacher_veto_can_require_active_teacher_conflict() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher_absent = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher_absent,
        feature_index={edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            require_teacher_conflict=True,
        ),
    )

    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "no_teacher_conflict"
    assert report.rows[0]["teacher_conflict"] == 0
    np.testing.assert_array_equal(report.tracks, predicted)


def test_teacher_veto_conflict_gate_accepts_competing_teacher_target() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher_conflict = np.asarray([[10, 11, 99, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher_conflict,
        feature_index={edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            require_teacher_conflict=True,
        ),
    )

    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_split_edge"
    assert report.rows[0]["teacher_conflict"] == 1


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


def test_teacher_veto_risk_order_with_max_applied_vetoes() -> None:
    predicted = np.asarray(
        [
            [10, 11, 12, -1],
            [20, 21, 22, -1],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [10, 11, -1, -1],
            [20, 21, -1, -1],
        ],
        dtype=int,
    )
    stronger_edge = (1, 2, 11, 12)
    weaker_edge = (1, 2, 21, 22)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={
            stronger_edge: _weak_feature(),
            weaker_edge: _weaker_feature(),
        },
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            edge_order="risk",
            max_applied_vetoes=1,
        ),
    )

    assert report.rows[0]["roi_a"] == 21
    assert report.rows[0]["roi_b"] == 22
    assert report.rows[0]["applied"] == 1
    assert report.rows[1]["reason"] == "max_applied_vetoes_reached"
    np.testing.assert_array_equal(report.tracks[0], [10, 11, 12, -1])


def test_teacher_veto_can_try_weakest_edge_first_with_cap() -> None:
    predicted = np.asarray(
        [
            [10, 11, 12, -1],
            [20, 21, 22, -1],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [10, 11, -1, -1],
            [20, 21, -1, -1],
        ],
        dtype=int,
    )
    stronger_edge = (1, 2, 11, 12)
    weaker_edge = (1, 2, 21, 22)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={
            stronger_edge: _feature_with_margin(
                threshold_margin=0.09, row_margin=0.15, column_margin=0.15
            ),
            weaker_edge: _feature_with_margin(
                threshold_margin=0.01, row_margin=0.02, column_margin=0.02
            ),
        },
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            veto_order="weakest",
            max_applied_vetoes=1,
        ),
    )

    applied_rows = [row for row in report.rows if int(row["applied"])]
    assert len(applied_rows) == 1
    assert applied_rows[0]["roi_a"] == 21
    assert applied_rows[0]["roi_b"] == 22


def test_teacher_veto_zero_cap_prevents_vetoes() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            max_applied_vetoes=0,
        ),
    )

    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "max_applied_vetoes_reached"
    np.testing.assert_array_equal(report.tracks, predicted)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_fragment_observations": 0}, "min_fragment_observations"),
        ({"max_applied_vetoes": -1}, "max_applied_vetoes"),
    ],
)
def test_teacher_veto_config_rejects_invalid_budget_values(
    kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        veto.TeacherVetoConfig(**kwargs)


def test_teacher_veto_rejects_centroid_too_close_when_requested() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            min_centroid_distance=5.0,
        ),
    )

    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "centroid_distance_too_low"
    np.testing.assert_array_equal(report.tracks, predicted)


def test_teacher_veto_rejects_area_ratio_too_high_when_requested() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            max_area_ratio=0.55,
        ),
    )

    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "area_ratio_too_high"
    np.testing.assert_array_equal(report.tracks, predicted)


def test_teacher_veto_accepts_geometrically_weak_edge_when_requested() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weaker_feature()},
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            min_centroid_distance=5.0,
            max_area_ratio=0.55,
        ),
    )

    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_split_edge"


def test_teacher_veto_complete_track_only_rejects_incomplete_row() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            allow_complete_track_veto=True,
            complete_track_veto_only=True,
            min_fragment_observations=1,
        ),
    )

    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "not_complete_track"
    np.testing.assert_array_equal(report.tracks, predicted)


def test_teacher_veto_complete_track_only_can_split_complete_row() -> None:
    predicted = np.asarray([[10, 11, 12, 13]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            allow_complete_track_veto=True,
            complete_track_veto_only=True,
            min_fragment_observations=2,
        ),
    )

    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_split_edge"
    np.testing.assert_array_equal(report.tracks, [[10, 11, -1, -1], [-1, -1, 12, 13]])


def test_teacher_veto_can_consider_complete_rows_absent_from_teacher() -> None:
    predicted = np.asarray([[10, 11, 12, 13]], dtype=int)
    teacher = np.asarray(
        [
            [10, 11, 12, -1],
            [-1, -1, 12, 13],
        ],
        dtype=int,
    )
    bridge_edge = (1, 2, 11, 12)

    default_report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={bridge_edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            allow_complete_track_veto=True,
            complete_track_veto_only=True,
            min_fragment_observations=2,
        ),
    )
    assert default_report.rows == ()

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={bridge_edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            allow_complete_track_veto=True,
            complete_track_veto_only=True,
            include_teacher_supported_complete_track_edges=True,
            min_fragment_observations=2,
        ),
    )

    applied_rows = [row for row in report.rows if int(row["applied"])]
    assert len(applied_rows) == 1
    assert applied_rows[0]["session_a"] == 1
    assert applied_rows[0]["session_b"] == 2
    assert applied_rows[0]["reason"] == "accepted_split_edge"
    assert applied_rows[0]["complete_teacher_row_supported"] == 0
    np.testing.assert_array_equal(report.tracks, [[10, 11, -1, -1], [-1, -1, 12, 13]])


def test_teacher_veto_does_not_split_teacher_supported_complete_row() -> None:
    predicted = np.asarray([[10, 11, 12, 13]], dtype=int)
    teacher = np.asarray([[10, 11, 12, 13]], dtype=int)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        config=veto.TeacherVetoConfig(
            allow_complete_track_veto=True,
            complete_track_veto_only=True,
            include_teacher_supported_complete_track_edges=True,
            min_fragment_observations=2,
        ),
    )

    assert report.rows == ()
    np.testing.assert_array_equal(report.tracks, predicted)


def test_teacher_veto_rejects_edges_that_fail_optional_shape_cell_gates() -> None:
    predicted = np.asarray(
        [
            [10, 11, 12, -1],
            [20, 21, 22, -1],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [10, 11, -1, -1],
            [20, 21, -1, -1],
        ],
        dtype=int,
    )
    low_risk_edge = (1, 2, 11, 12)
    high_risk_edge = (1, 2, 21, 22)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={
            low_risk_edge: _weak_feature(),
            high_risk_edge: _weaker_feature(),
        },
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            min_centroid_distance=5.0,
            max_area_ratio=0.55,
            max_cell_probability=0.50,
        ),
    )

    rejected = next(row for row in report.rows if row["roi_a"] == 11)
    applied = next(row for row in report.rows if row["roi_a"] == 21)
    assert rejected["applied"] == 0
    assert rejected["reason"] == "centroid_distance_too_low"
    assert applied["applied"] == 1
    assert applied["reason"] == "accepted_split_edge"
    assert applied["min_cell_probability"] == 0.35


def test_teacher_veto_can_require_non_hungarian_edges() -> None:
    predicted = np.asarray([[10, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)
    edge = (1, 2, 11, 12)

    report = veto.apply_teacher_veto_edges(
        predicted,
        teacher,
        feature_index={edge: _weak_feature()},
        config=veto.TeacherVetoConfig(
            min_fragment_observations=1,
            require_unassigned_by_hungarian=True,
        ),
    )

    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "assigned_by_hungarian"


def test_teacher_veto_parser_exposes_order_and_cap() -> None:
    args = veto.build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--teacher-veto-edge-order",
            "risk",
            "--veto-order",
            "weakest",
            "--max-applied-vetoes",
            "1",
            "--min-centroid-distance",
            "5",
            "--max-area-ratio",
            "0.55",
            "--max-cell-probability",
            "0.5",
            "--require-unassigned-by-hungarian",
            "--require-teacher-conflict",
            "--complete-track-veto-only",
            "--include-teacher-supported-complete-track-edges",
        ]
    )

    assert args.teacher_veto_edge_order == "risk"
    assert args.veto_order == "weakest"
    assert args.max_applied_vetoes == 1
    assert args.min_centroid_distance == 5
    assert args.max_area_ratio == 0.55
    assert args.max_cell_probability == 0.5
    assert args.require_unassigned_by_hungarian is True
    assert args.require_teacher_conflict is True
    assert args.complete_track_veto_only is True
    assert args.include_teacher_supported_complete_track_edges is True


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--min-veto-fragment-observations", "0"),
        ("--max-applied-vetoes", "-1"),
    ],
)
def test_teacher_veto_parser_rejects_invalid_budget_args(
    option: str, value: str
) -> None:
    with pytest.raises(SystemExit):
        veto.build_arg_parser().parse_args(["--data", "track2p-root", option, value])
