from __future__ import annotations

import numpy as np
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    ResidualFeature,
)
from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
    TeacherEdgeFeatureGate,
    _teacher_completion_gate_kwargs,
    _teacher_edge_order_requires_feature_index,
    _teacher_edge_order_uses_confidence_features,
    apply_teacher_adjacent_rescue_edges,
    build_arg_parser,
    teacher_adjacent_repair_preset_kwargs,
    teacher_feature_gate_from_preset,
)


def test_teacher_completion_gate_kwargs_preserve_exact_aliases() -> None:
    kwargs = _teacher_completion_gate_kwargs(
        allow_teacher_complete_row_rescue=True,
        allow_teacher_supported_completion=False,
        allow_teacher_supported_completing_rescue=False,
        allow_teacher_confirmed_completing_rescue=False,
    )

    assert kwargs == {
        "allow_teacher_complete_row_rescue": True,
        "allow_teacher_supported_completion": False,
        "allow_teacher_supported_completing_rescue": False,
        "allow_teacher_confirmed_completing_rescue": False,
    }


def test_teacher_edge_order_requires_feature_index_for_seed_confidence() -> None:
    assert _teacher_edge_order_requires_feature_index("confidence")
    assert _teacher_edge_order_requires_feature_index("cell-confidence")
    assert _teacher_edge_order_requires_feature_index("dynamic-confidence")
    assert _teacher_edge_order_requires_feature_index("dynamic-cell-confidence")
    assert _teacher_edge_order_requires_feature_index("dynamic-seed-confidence")
    assert _teacher_edge_order_requires_feature_index("dynamic-seed-cell-confidence")

    assert not _teacher_edge_order_requires_feature_index("lexicographic")
    assert not _teacher_edge_order_requires_feature_index("structural")
    assert not _teacher_edge_order_requires_feature_index("dynamic-structural")


def test_teacher_edge_order_uses_confidence_features() -> None:
    assert _teacher_edge_order_uses_confidence_features("confidence")
    assert _teacher_edge_order_uses_confidence_features("cell-confidence")
    assert _teacher_edge_order_uses_confidence_features("dynamic-confidence")
    assert _teacher_edge_order_uses_confidence_features("dynamic-cell-confidence")
    assert _teacher_edge_order_uses_confidence_features("dynamic-seed-confidence")
    assert _teacher_edge_order_uses_confidence_features("dynamic-seed-cell-confidence")

    assert not _teacher_edge_order_uses_confidence_features("structural")
    assert not _teacher_edge_order_uses_confidence_features("dynamic-structural")


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


def test_teacher_adjacent_rescue_allows_seed_only_source_backfill() -> None:
    predicted = np.asarray([[-1, 20, -1]], dtype=int)
    teacher = np.asarray([[10, 20, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_source_backfill=False,
        allow_seed_source_backfill=True,
    )

    np.testing.assert_array_equal(output.tracks, [[10, 20, -1]])
    assert output.rows == (
        {
            "session_a": 0,
            "session_b": 1,
            "roi_a": 10,
            "roi_b": 20,
            "applied": 1,
            "reason": "accepted_insert_source",
            "source_row": -1,
            "target_row": 0,
            "teacher_complete_row_supported": 0,
            "occurrence_index": 0,
        },
    )


def test_teacher_adjacent_rescue_seed_source_backfill_can_bypass_generic_source_backfill() -> (
    None
):
    predicted = np.asarray([[-1, 20, 30]], dtype=int)
    teacher = np.asarray([[10, 20, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_source_backfill=False,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
    )

    np.testing.assert_array_equal(output.tracks, [[10, 20, 30]])
    assert output.rows[0]["applied"] == 1
    assert output.rows[0]["reason"] == "accepted_insert_source"


def test_teacher_adjacent_rescue_seed_source_only_does_not_allow_nonseed_backfill() -> (
    None
):
    predicted = np.asarray([[10, -1, 30]], dtype=int)
    teacher = np.asarray([[-1, 20, 30]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_source_backfill=False,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
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


def test_dynamic_seed_confidence_prioritizes_missing_seed_source_backfill() -> None:
    predicted = np.asarray(
        [
            [-1, 11, -1, 40],
            [50, -1, 70, 80],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [-1, -1, 30, 40],
            [-1, 60, 70, -1],
        ],
        dtype=int,
    )
    edge_feature_index = {
        (2, 3, 30, 40): ResidualFeature(
            registered_iou=0.10,
            centroid_distance=4.0,
            area_ratio=0.60,
            threshold_margin=0.05,
            assigned_by_hungarian=1,
        ),
        (1, 2, 60, 70): ResidualFeature(
            registered_iou=0.90,
            centroid_distance=1.0,
            area_ratio=0.95,
            threshold_margin=0.50,
            assigned_by_hungarian=1,
        ),
    }

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=2,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
        edge_order="dynamic-seed-confidence",
        edge_feature_index=edge_feature_index,
        max_applied_edits=1,
    )

    np.testing.assert_array_equal(output.tracks[0], [-1, 11, 30, 40])
    assert output.rows[0]["session_a"] == 2
    assert output.rows[0]["roi_a"] == 30
    assert output.rows[0]["applied"] == 1
    assert output.rows[0]["reason"] == "accepted_insert_source"


def test_cell_confidence_order_prioritizes_cell_probability_before_iou() -> None:
    predicted = np.asarray([[10, -1, -1], [11, -1, -1]], dtype=int)
    teacher = np.asarray([[10, 20, -1], [11, 21, -1]], dtype=int)
    edge_feature_index = {
        (0, 1, 10, 20): ResidualFeature(
            registered_iou=0.90,
            centroid_distance=1.0,
            area_ratio=0.95,
            threshold_margin=0.50,
            assigned_by_hungarian=1,
            cell_probability_a=0.55,
            cell_probability_b=0.55,
        ),
        (0, 1, 11, 21): ResidualFeature(
            registered_iou=0.20,
            centroid_distance=2.0,
            area_ratio=0.80,
            threshold_margin=0.10,
            assigned_by_hungarian=1,
            cell_probability_a=0.95,
            cell_probability_b=0.95,
        ),
    }

    iou_ordered = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        edge_order="confidence",
        edge_feature_index=edge_feature_index,
        max_applied_edits=1,
    )
    cell_ordered = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        edge_order="cell-confidence",
        edge_feature_index=edge_feature_index,
        max_applied_edits=1,
    )

    np.testing.assert_array_equal(iou_ordered.tracks, [[10, 20, -1], [11, -1, -1]])
    np.testing.assert_array_equal(cell_ordered.tracks, [[10, -1, -1], [11, 21, -1]])
    assert iou_ordered.rows[0]["roi_b"] == 20
    assert cell_ordered.rows[0]["roi_b"] == 21


def test_teacher_adjacent_rescue_can_filter_to_seed_source_backfills() -> None:
    predicted = np.asarray(
        [
            [10, -1, -1, -1],
            [-1, 21, 22, -1],
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

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        teacher_action_filter="seed-source-backfill",
        edge_order="lexicographic",
    )

    np.testing.assert_array_equal(output.tracks, [[10, -1, -1, -1], [20, 21, 22, -1]])
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "action_filter_seed-source-backfill"
    assert output.rows[1]["applied"] == 1
    assert output.rows[1]["reason"] == "accepted_insert_source"


def test_teacher_adjacent_rescue_can_filter_to_target_or_seed_source_union() -> None:
    predicted = np.asarray(
        [
            [10, -1, -1],
            [-1, 21, -1],
            [30, -1, -1],
            [-1, 31, -1],
            [-1, -1, 42],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [10, 11, -1],
            [20, 21, -1],
            [30, 31, -1],
            [-1, 41, 42],
        ],
        dtype=int,
    )

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
        teacher_action_filter="target-extension-or-seed-source-backfill",
        edge_order="lexicographic",
    )

    np.testing.assert_array_equal(
        output.tracks,
        [
            [10, 11, -1],
            [20, 21, -1],
            [30, -1, -1],
            [-1, 31, -1],
            [-1, -1, 42],
        ],
    )
    applied_reasons = [row["reason"] for row in output.rows if int(row["applied"])]
    rejected_reasons = [row["reason"] for row in output.rows if not int(row["applied"])]
    assert applied_reasons == ["accepted_insert_target", "accepted_insert_source"]
    assert "action_filter_target-extension-or-seed-source-backfill" in rejected_reasons


def test_teacher_adjacent_rescue_can_filter_to_completing_rescues() -> None:
    predicted = np.asarray(
        [
            [10, -1, -1],
            [20, 21, -1],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [10, 11, -1],
            [20, 21, 22],
        ],
        dtype=int,
    )

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_completing_rescue=True,
        teacher_action_filter="completing-rescue",
        edge_order="lexicographic",
    )

    np.testing.assert_array_equal(output.tracks, [[10, -1, -1], [20, 21, 22]])
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "action_filter_completing-rescue"
    assert output.rows[1]["applied"] == 1
    assert output.rows[1]["reason"] == "accepted_insert_target"


def test_dynamic_teacher_rescue_reconsiders_action_filter_after_edit() -> None:
    predicted = np.asarray([[10, -1, -1]], dtype=int)
    teacher = np.asarray([[10, 20, 30]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_completing_rescue=True,
        teacher_action_filter="target-extension",
        edge_order="dynamic-structural",
        max_applied_edits=2,
    )

    np.testing.assert_array_equal(output.tracks, [[10, 20, 30]])
    applied_edges = [
        (row["session_a"], row["session_b"], row["roi_a"], row["roi_b"])
        for row in output.rows
        if int(row["applied"])
    ]
    assert applied_edges == [(0, 1, 10, 20), (1, 2, 20, 30)]


def test_teacher_adjacent_rescue_can_filter_to_completing_seed_source_backfills() -> (
    None
):
    predicted = np.asarray(
        [
            [-1, 20, 30],
            [-1, 21, -1],
            [40, -1, 42],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [10, 20, -1],
            [11, 21, -1],
            [40, 41, 42],
        ],
        dtype=int,
    )

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_source_backfill=False,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
        allow_completing_rescue=True,
        teacher_action_filter="completing-seed-source-backfill",
        edge_order="lexicographic",
    )

    np.testing.assert_array_equal(
        output.tracks, [[10, 20, 30], [-1, 21, -1], [40, -1, 42]]
    )
    assert [row["reason"] for row in output.rows if int(row["applied"])] == [
        "accepted_insert_source"
    ]
    assert all(
        row["reason"] == "action_filter_completing-seed-source-backfill"
        for row in output.rows
        if not int(row["applied"])
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


def test_teacher_repair_preset_targets_missing_seed_source_backfill() -> None:
    kwargs = teacher_adjacent_repair_preset_kwargs("missing-seed-high-confidence")

    assert kwargs == {
        "allow_source_backfill": False,
        "allow_seed_source_backfill": True,
        "allow_completing_seed_source_backfill": True,
        "teacher_edge_order": "dynamic-seed-confidence",
        "teacher_action_filter": "seed-source-backfill",
        "teacher_feature_preset": "seed-source-high-confidence",
        "min_component_observations": 2,
        "max_applied_edits": 2,
    }


def test_teacher_cell_confident_repair_preset_targets_missing_seed_source_backfill() -> (
    None
):
    kwargs = teacher_adjacent_repair_preset_kwargs("missing-seed-cell-confident")

    assert kwargs == {
        "allow_source_backfill": False,
        "allow_seed_source_backfill": True,
        "allow_completing_seed_source_backfill": True,
        "teacher_edge_order": "dynamic-seed-confidence",
        "teacher_action_filter": "seed-source-backfill",
        "teacher_feature_preset": "seed-source-cell-confident",
        "min_component_observations": 2,
        "max_applied_edits": 3,
    }


def test_teacher_moderate_iou_repair_preset_targets_missing_seed_source_backfill() -> (
    None
):
    kwargs = teacher_adjacent_repair_preset_kwargs("missing-seed-moderate-iou")

    assert kwargs == {
        "allow_source_backfill": False,
        "allow_seed_source_backfill": True,
        "allow_completing_seed_source_backfill": True,
        "teacher_edge_order": "dynamic-seed-confidence",
        "teacher_action_filter": "seed-source-backfill",
        "teacher_feature_preset": "seed-source-moderate-iou",
        "min_component_observations": 2,
        "max_applied_edits": 2,
    }


def test_teacher_completing_seed_source_preset_targets_only_completing_backfills() -> (
    None
):
    kwargs = teacher_adjacent_repair_preset_kwargs(
        "missing-seed-completing-moderate-iou"
    )

    assert kwargs == {
        "allow_source_backfill": False,
        "allow_seed_source_backfill": True,
        "allow_completing_seed_source_backfill": True,
        "allow_fragment_merges": False,
        "teacher_edge_order": "dynamic-seed-cell-confidence",
        "teacher_action_filter": "completing-seed-source-backfill",
        "teacher_feature_preset": "seed-source-moderate-iou",
        "min_component_observations": 2,
        "max_applied_edits": 1,
    }


def test_track2p_fn_repair_preset_targets_teacher_extensions() -> None:
    kwargs = teacher_adjacent_repair_preset_kwargs("track2p-fn-high-confidence")

    assert kwargs == {
        "teacher_action_filter": "target-extension",
        "teacher_edge_order": "dynamic-confidence",
        "teacher_feature_preset": "track2p-fn-rescue",
        "min_component_observations": 2,
        "max_applied_edits": 3,
    }


def test_track2p_fn_moderate_iou_repair_preset_targets_cell_gated_extensions() -> None:
    kwargs = teacher_adjacent_repair_preset_kwargs(
        "track2p-fn-moderate-iou-cell-confident"
    )

    assert kwargs == {
        "teacher_action_filter": "target-extension",
        "teacher_edge_order": "dynamic-cell-confidence",
        "teacher_feature_preset": "moderate-iou-cell-confidence",
        "min_component_observations": 2,
        "max_applied_edits": 3,
    }


def test_residual_union_repair_preset_targets_two_residual_buckets() -> None:
    kwargs = teacher_adjacent_repair_preset_kwargs("residual-union-cell-confident")

    assert kwargs == {
        "allow_source_backfill": False,
        "allow_seed_source_backfill": True,
        "allow_completing_seed_source_backfill": True,
        "allow_fragment_merges": False,
        "teacher_action_filter": "target-extension-or-seed-source-backfill",
        "teacher_edge_order": "dynamic-seed-cell-confidence",
        "teacher_feature_preset": "residual-fn-cell-confident",
        "min_component_observations": 2,
        "max_applied_edits": 3,
    }


def test_residual_union_action_specific_preset_splits_feature_gates() -> None:
    kwargs = teacher_adjacent_repair_preset_kwargs("residual-union-action-specific")

    assert kwargs == {
        "allow_source_backfill": False,
        "allow_seed_source_backfill": True,
        "allow_completing_seed_source_backfill": True,
        "allow_fragment_merges": False,
        "teacher_action_filter": "target-extension-or-seed-source-backfill",
        "teacher_edge_order": "dynamic-seed-cell-confidence",
        "teacher_feature_preset": "none",
        "target_extension_feature_preset": "moderate-iou-cell-confidence",
        "seed_source_feature_preset": "seed-source-cell-confident",
        "min_component_observations": 2,
        "max_applied_edits": 3,
    }


def test_completing_rescue_action_specific_preset_targets_complete_rows() -> None:
    kwargs = teacher_adjacent_repair_preset_kwargs("completing-rescue-action-specific")

    assert kwargs == {
        "allow_teacher_complete_row_rescue": True,
        "allow_completing_rescue": True,
        "allow_source_backfill": False,
        "allow_seed_source_backfill": True,
        "allow_completing_seed_source_backfill": True,
        "allow_fragment_merges": False,
        "teacher_action_filter": "completing-rescue",
        "teacher_edge_order": "dynamic-seed-cell-confidence",
        "teacher_feature_preset": "none",
        "target_extension_feature_preset": "moderate-iou-cell-confidence",
        "seed_source_feature_preset": "seed-source-cell-confident",
        "min_component_observations": 2,
        "max_applied_edits": 2,
    }

    assert (
        teacher_adjacent_repair_preset_kwargs("complete-row-rescue-action-specific")
        == kwargs
    )
    assert (
        teacher_adjacent_repair_preset_kwargs("complete-row-action-specific") == kwargs
    )


def test_residual_union_action_balanced_preset_adds_action_caps() -> None:
    kwargs = teacher_adjacent_repair_preset_kwargs("residual-union-action-balanced")

    assert kwargs == {
        "allow_source_backfill": False,
        "allow_seed_source_backfill": True,
        "allow_completing_seed_source_backfill": True,
        "allow_fragment_merges": False,
        "teacher_action_filter": "target-extension-or-seed-source-backfill",
        "teacher_edge_order": "dynamic-seed-cell-confidence",
        "teacher_feature_preset": "none",
        "target_extension_feature_preset": "moderate-iou-cell-confidence",
        "seed_source_feature_preset": "seed-source-cell-confident",
        "min_component_observations": 2,
        "max_applied_edits": 3,
        "max_seed_source_backfill_edits": 1,
        "max_target_extension_edits": 2,
    }

    assert teacher_adjacent_repair_preset_kwargs("residual-union-balanced") == kwargs


def test_teacher_rescue_parser_accepts_completing_rescue_preset() -> None:
    args = build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--teacher-repair-preset",
            "completing-rescue-action-specific",
        ]
    )

    assert args.teacher_repair_preset == "completing-rescue-action-specific"


def test_completing_rescue_action_specific_preset_applies_confirmed_completion() -> (
    None
):
    predicted = np.asarray([[100, 20, 30, -1]], dtype=int)
    teacher = np.asarray([[100, 20, 30, 40]], dtype=int)
    kwargs = teacher_adjacent_repair_preset_kwargs("completing-rescue-action-specific")
    target_gate = teacher_feature_gate_from_preset(
        str(kwargs["target_extension_feature_preset"])
    )
    seed_gate = teacher_feature_gate_from_preset(
        str(kwargs["seed_source_feature_preset"])
    )

    report = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_teacher_complete_row_rescue=bool(
            kwargs["allow_teacher_complete_row_rescue"]
        ),
        allow_fragment_merges=bool(kwargs["allow_fragment_merges"]),
        teacher_action_filter=str(kwargs["teacher_action_filter"]),
        edge_order=str(kwargs["teacher_edge_order"]),
        target_extension_feature_gate=target_gate,
        seed_source_feature_gate=seed_gate,
        edge_feature_index={
            (2, 3, 30, 40): ResidualFeature(
                registered_iou=0.25,
                centroid_distance=2.0,
                area_ratio=0.80,
                cell_probability_a=0.92,
                cell_probability_b=0.91,
                threshold_margin=0.10,
            )
        },
        min_component_observations=int(kwargs["min_component_observations"]),
        max_applied_edits=int(kwargs["max_applied_edits"]),
    )

    np.testing.assert_array_equal(report.tracks, [[100, 20, 30, 40]])
    applied_rows = tuple(row for row in report.rows if int(row["applied"]))
    assert len(applied_rows) == 1
    assert applied_rows[0]["reason"] == "accepted_insert_target"
    assert applied_rows[0]["teacher_complete_row_supported"] == 1


def test_seed_source_high_confidence_preset_accepts_seed_backfill_without_hungarian() -> (
    None
):
    predicted = np.asarray([[-1, 20, 30]], dtype=int)
    teacher = np.asarray([[10, 20, -1]], dtype=int)
    gate = teacher_feature_gate_from_preset("seed-source-high-confidence")

    assert gate is not None
    assert not gate.require_hungarian

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
        feature_gate=gate,
        edge_feature_index={
            (0, 1, 10, 20): ResidualFeature(
                registered_iou=0.18,
                centroid_distance=5.5,
                area_ratio=0.65,
                cell_probability_a=0.75,
                cell_probability_b=0.80,
                assigned_by_hungarian=0,
            )
        },
    )

    np.testing.assert_array_equal(output.tracks, [[10, 20, 30]])
    assert output.rows[0]["applied"] == 1
    assert output.rows[0]["reason"] == "accepted_insert_source"


def test_seed_source_cell_confident_preset_allows_low_iou_seed_backfill() -> None:
    predicted = np.asarray([[-1, 20, 30]], dtype=int)
    teacher = np.asarray([[10, 20, -1]], dtype=int)
    gate = teacher_feature_gate_from_preset("seed-source-cell-confident")

    assert gate is not None
    assert gate.min_registered_iou == 0.0
    assert gate.min_cell_probability == 0.85
    assert not gate.require_hungarian

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
        teacher_action_filter="seed-source-backfill",
        feature_gate=gate,
        edge_feature_index={
            (0, 1, 10, 20): ResidualFeature(
                registered_iou=0.02,
                centroid_distance=5.0,
                area_ratio=0.65,
                cell_probability_a=0.90,
                cell_probability_b=0.86,
                assigned_by_hungarian=0,
            )
        },
    )

    np.testing.assert_array_equal(output.tracks, [[10, 20, 30]])
    assert output.rows[0]["applied"] == 1
    assert output.rows[0]["reason"] == "accepted_insert_source"


def test_seed_source_cell_confident_preset_rejects_low_cell_seed_backfill() -> None:
    predicted = np.asarray([[-1, 20, 30]], dtype=int)
    teacher = np.asarray([[10, 20, -1]], dtype=int)
    gate = teacher_feature_gate_from_preset("seed-source-cell-confident")

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
        teacher_action_filter="seed-source-backfill",
        feature_gate=gate,
        edge_feature_index={
            (0, 1, 10, 20): ResidualFeature(
                registered_iou=0.20,
                centroid_distance=2.0,
                area_ratio=0.90,
                cell_probability_a=0.84,
                cell_probability_b=0.95,
                assigned_by_hungarian=0,
            )
        },
    )

    np.testing.assert_array_equal(output.tracks, predicted)
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "feature_gate_cell_probability"


def test_action_specific_feature_gates_separate_target_and_seed_backfill() -> None:
    predicted = np.asarray(
        [
            [10, -1, -1],
            [-1, 21, 22],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [10, 11, -1],
            [20, 21, -1],
        ],
        dtype=int,
    )
    edge_feature_index = {
        (0, 1, 10, 11): ResidualFeature(
            registered_iou=0.20,
            centroid_distance=2.0,
            area_ratio=0.80,
            cell_probability_a=0.60,
            cell_probability_b=0.60,
        ),
        (0, 1, 20, 21): ResidualFeature(
            registered_iou=0.01,
            centroid_distance=3.0,
            area_ratio=0.65,
            cell_probability_a=0.88,
            cell_probability_b=0.90,
        ),
    }

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_source_backfill=False,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
        teacher_action_filter="target-extension-or-seed-source-backfill",
        edge_order="lexicographic",
        edge_feature_index=edge_feature_index,
        target_extension_feature_gate=TeacherEdgeFeatureGate(min_cell_probability=0.90),
        seed_source_feature_gate=TeacherEdgeFeatureGate(min_cell_probability=0.85),
    )

    np.testing.assert_array_equal(output.tracks, [[10, -1, -1], [20, 21, 22]])
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "feature_gate_cell_probability"
    assert output.rows[1]["applied"] == 1
    assert output.rows[1]["reason"] == "accepted_insert_source"


def test_seed_source_moderate_iou_preset_rejects_high_iou_teacher_edge() -> None:
    predicted = np.asarray([[-1, 20, 30]], dtype=int)
    teacher = np.asarray([[10, 20, -1]], dtype=int)
    gate = teacher_feature_gate_from_preset("seed-source-moderate-iou")

    assert gate is not None
    assert gate.max_registered_iou == 0.55
    assert not gate.require_hungarian

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
        teacher_action_filter="seed-source-backfill",
        feature_gate=gate,
        edge_feature_index={
            (0, 1, 10, 20): ResidualFeature(
                registered_iou=0.80,
                centroid_distance=2.0,
                area_ratio=0.90,
                cell_probability_a=0.90,
                cell_probability_b=0.95,
                threshold_margin=0.10,
                row_margin=0.20,
                column_margin=0.20,
                assigned_by_hungarian=0,
            )
        },
    )

    np.testing.assert_array_equal(output.tracks, predicted)
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "feature_gate_max_registered_iou"


def test_residual_fn_cell_confident_preset_requires_cells_without_hungarian() -> None:
    predicted = np.asarray([[10, -1, 30, -1]], dtype=int)
    teacher = np.asarray([[10, 20, -1, -1]], dtype=int)
    gate = teacher_feature_gate_from_preset("residual-fn-cell-confident")

    assert gate is not None
    assert gate.min_registered_iou == 0.10
    assert gate.max_centroid_distance == 6.0
    assert gate.min_area_ratio == 0.45
    assert gate.min_cell_probability == 0.80
    assert not gate.require_hungarian

    low_cell_report = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        feature_gate=gate,
        edge_feature_index={
            (0, 1, 10, 20): ResidualFeature(
                registered_iou=0.12,
                centroid_distance=5.0,
                area_ratio=0.50,
                cell_probability_a=0.79,
                cell_probability_b=0.90,
                assigned_by_hungarian=0,
            )
        },
    )

    np.testing.assert_array_equal(low_cell_report.tracks, predicted)
    assert low_cell_report.rows[0]["applied"] == 0
    assert low_cell_report.rows[0]["reason"] == "feature_gate_cell_probability"

    high_cell_report = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        feature_gate=gate,
        edge_feature_index={
            (0, 1, 10, 20): ResidualFeature(
                registered_iou=0.12,
                centroid_distance=5.0,
                area_ratio=0.50,
                cell_probability_a=0.80,
                cell_probability_b=0.90,
                assigned_by_hungarian=0,
            )
        },
    )

    np.testing.assert_array_equal(high_cell_report.tracks, [[10, 20, 30, -1]])
    assert high_cell_report.rows[0]["applied"] == 1
    assert high_cell_report.rows[0]["reason"] == "accepted_insert_target"


def test_residual_union_preset_spends_tiny_budget_on_cell_confident_seed_edge() -> None:
    predicted = np.asarray([[-1, 20, 22], [-1, 21, 23]], dtype=int)
    teacher = np.asarray([[10, 20, -1], [11, 21, -1]], dtype=int)
    kwargs = teacher_adjacent_repair_preset_kwargs("residual-union-cell-confident")
    gate = teacher_feature_gate_from_preset(str(kwargs["teacher_feature_preset"]))

    report = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_source_backfill=bool(kwargs["allow_source_backfill"]),
        allow_seed_source_backfill=bool(kwargs["allow_seed_source_backfill"]),
        allow_completing_seed_source_backfill=bool(
            kwargs["allow_completing_seed_source_backfill"]
        ),
        allow_fragment_merges=bool(kwargs["allow_fragment_merges"]),
        teacher_action_filter=str(kwargs["teacher_action_filter"]),
        edge_order=str(kwargs["teacher_edge_order"]),
        feature_gate=gate,
        edge_feature_index={
            (0, 1, 10, 20): ResidualFeature(
                registered_iou=0.90,
                centroid_distance=1.0,
                area_ratio=0.95,
                cell_probability_a=0.81,
                cell_probability_b=0.81,
            ),
            (0, 1, 11, 21): ResidualFeature(
                registered_iou=0.20,
                centroid_distance=2.0,
                area_ratio=0.90,
                cell_probability_a=0.97,
                cell_probability_b=0.97,
            ),
        },
        max_applied_edits=int(kwargs["max_applied_edits"]),
    )

    np.testing.assert_array_equal(report.tracks, [[10, 20, 22], [11, 21, 23]])
    assert report.rows[0]["roi_a"] == 11
    assert report.rows[0]["applied"] == 1


def test_teacher_adjacent_rescue_enforces_per_action_edit_caps() -> None:
    predicted = np.asarray(
        [
            [-1, 10, -1],
            [-1, 20, -1],
            [30, -1, -1],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [1, 10, -1],
            [2, 20, -1],
            [30, 31, -1],
        ],
        dtype=int,
    )

    report = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_source_backfill=False,
        allow_seed_source_backfill=True,
        teacher_action_filter="target-extension-or-seed-source-backfill",
        edge_order="dynamic-seed-confidence",
        max_applied_edits=3,
        max_seed_source_backfill_edits=1,
        max_target_extension_edits=1,
    )

    np.testing.assert_array_equal(
        report.tracks,
        [
            [1, 10, -1],
            [-1, 20, -1],
            [30, 31, -1],
        ],
    )
    applied_edges = [
        (row["session_a"], row["session_b"], row["roi_a"], row["roi_b"])
        for row in report.rows
        if int(row["applied"])
    ]
    assert applied_edges == [(0, 1, 1, 10), (0, 1, 30, 31)]
    assert any(
        row["reason"] == "max_seed_source_backfill_edits_reached" for row in report.rows
    )


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


def test_teacher_adjacent_rescue_exact_completion_rejects_partial_teacher_row() -> None:
    predicted = np.asarray([[100, 20, -1, 40]], dtype=int)
    partial_teacher = np.asarray([[100, 20, 30, -1]], dtype=int)

    exact = apply_teacher_adjacent_rescue_edges(
        predicted,
        partial_teacher,
        seed_session=0,
        allow_teacher_complete_row_rescue=True,
    )

    np.testing.assert_array_equal(exact.tracks, predicted)
    assert exact.rows[-1]["applied"] == 0
    assert exact.rows[-1]["reason"] == "would_complete_track"
    assert exact.rows[-1]["teacher_complete_row_supported"] == 0

    supported = apply_teacher_adjacent_rescue_edges(
        predicted,
        partial_teacher,
        seed_session=0,
        allow_teacher_supported_completion=True,
    )

    np.testing.assert_array_equal(supported.tracks, [[100, 20, 30, 40]])
    assert supported.rows[-1]["applied"] == 1
    assert supported.rows[-1]["teacher_complete_row_supported"] == 1


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


def test_dynamic_teacher_rescue_revisits_action_filter_after_insert() -> None:
    """A deferred teacher edge can become a target-extension after an insert.

    The dynamic rescue loop is supposed to recompute structural eligibility after
    accepted edits.  If action-filter rejections are treated as permanent, the
    second teacher edge below is rejected as ``other`` before the first edge
    inserts its source ROI, and the chain cannot grow.  The desired behavior is to
    defer that rejection, apply the first edge, then revisit and apply the second.
    """

    predicted = np.asarray([[100, -1, -1, -1]], dtype=int)
    teacher = np.asarray([[100, 200, 300, -1]], dtype=int)

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        teacher_action_filter="target-extension",
        edge_order="dynamic-structural",
    )

    np.testing.assert_array_equal(output.tracks, [[100, 200, 300, -1]])
    assert [row["reason"] for row in output.rows if int(row["applied"])] == [
        "accepted_insert_target",
        "accepted_insert_target",
    ]


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


def test_teacher_adjacent_rescue_feature_gate_rejects_low_cell_probability() -> None:
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
                cell_probability_a=0.95,
                cell_probability_b=0.40,
                threshold_margin=0.30,
                assigned_by_hungarian=1,
            ),
            (0, 1, 10, 12): ResidualFeature(
                registered_iou=0.8,
                centroid_distance=1.0,
                area_ratio=0.95,
                cell_probability_a=0.95,
                cell_probability_b=0.90,
                threshold_margin=0.30,
                assigned_by_hungarian=1,
            ),
        },
        feature_gate=TeacherEdgeFeatureGate(min_cell_probability=0.8),
    )

    np.testing.assert_array_equal(output.tracks, [[10, 12, -1]])
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "feature_gate_cell_probability"
    assert output.rows[1]["applied"] == 1
    assert output.rows[1]["reason"] == "accepted_insert_target"


def test_teacher_fn_rescue_preset_accepts_non_hungarian_track2p_edge() -> None:
    predicted = np.asarray([[10, -1, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1]], dtype=int)
    gate = teacher_feature_gate_from_preset("track2p-fn-rescue")

    assert gate is not None
    assert not gate.require_hungarian

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        feature_gate=gate,
        edge_feature_index={
            (0, 1, 10, 11): ResidualFeature(
                registered_iou=0.15,
                centroid_distance=5.5,
                area_ratio=0.50,
                cell_probability_a=0.65,
                cell_probability_b=0.66,
                assigned_by_hungarian=0,
            )
        },
    )

    np.testing.assert_array_equal(output.tracks, [[10, 11, -1]])
    assert output.rows[0]["applied"] == 1
    assert output.rows[0]["reason"] == "accepted_insert_target"


def test_teacher_fn_rescue_preset_rejects_low_cell_probability() -> None:
    predicted = np.asarray([[10, -1, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1]], dtype=int)
    gate = teacher_feature_gate_from_preset("track2p-fn-rescue")

    output = apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        feature_gate=gate,
        edge_feature_index={
            (0, 1, 10, 11): ResidualFeature(
                registered_iou=0.30,
                centroid_distance=2.0,
                area_ratio=0.80,
                cell_probability_a=0.95,
                cell_probability_b=0.40,
                assigned_by_hungarian=1,
            )
        },
    )

    np.testing.assert_array_equal(output.tracks, predicted)
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "feature_gate_cell_probability"


def test_teacher_adjacent_rescue_cell_high_confidence_preset_uses_cell_probability() -> (
    None
):
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
                cell_probability_a=0.95,
                cell_probability_b=0.40,
                row_margin=0.20,
                column_margin=0.20,
                threshold_margin=0.30,
                assigned_by_hungarian=1,
            ),
            (0, 1, 10, 12): ResidualFeature(
                registered_iou=0.8,
                centroid_distance=1.0,
                area_ratio=0.95,
                cell_probability_a=0.95,
                cell_probability_b=0.90,
                row_margin=0.20,
                column_margin=0.20,
                threshold_margin=0.30,
                assigned_by_hungarian=1,
            ),
        },
        feature_gate=teacher_feature_gate_from_preset("cell-high-confidence"),
    )

    np.testing.assert_array_equal(output.tracks, [[10, 12, -1]])
    assert output.rows[0]["applied"] == 0
    assert output.rows[0]["reason"] == "feature_gate_cell_probability"
    assert output.rows[1]["applied"] == 1
    assert output.rows[1]["reason"] == "accepted_insert_target"


def test_teacher_adjacent_rescue_cell_confident_preset_requires_cells() -> None:
    predicted = np.asarray([[10, -1, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1], [10, 12, -1]], dtype=int)
    gate = teacher_feature_gate_from_preset("cell-confident")

    assert gate is not None
    assert gate.min_cell_probability == 0.80
    assert gate.require_hungarian

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
                cell_probability_a=0.95,
                cell_probability_b=0.40,
                row_margin=0.3,
                column_margin=0.3,
                threshold_margin=0.30,
                assigned_by_hungarian=1,
            ),
            (0, 1, 10, 12): ResidualFeature(
                registered_iou=0.8,
                centroid_distance=1.0,
                area_ratio=0.95,
                cell_probability_a=0.95,
                cell_probability_b=0.90,
                row_margin=0.3,
                column_margin=0.3,
                threshold_margin=0.30,
                assigned_by_hungarian=1,
            ),
        },
        feature_gate=gate,
    )

    np.testing.assert_array_equal(output.tracks, [[10, 12, -1]])
    assert output.rows[0]["reason"] == "feature_gate_cell_probability"
    assert output.rows[1]["reason"] == "accepted_insert_target"
