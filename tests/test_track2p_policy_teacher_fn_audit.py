from __future__ import annotations

import numpy as np
from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_teacher_adjacent_rescue as rescue
from bayescatrack.experiments import track2p_policy_teacher_fn_audit as audit


def test_teacher_fn_audit_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-teacher-fn-audit"]

    assert canonical == "track2p-policy-teacher-fn-audit"
    assert cli._BENCHMARK_ALIASES["track2p-component-teacher-fn-audit"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_teacher_fn_audit"
    )


def test_teacher_adjacent_rescue_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-teacher-adjacent-rescue"]

    assert canonical == "track2p-policy-teacher-adjacent-rescue"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-teacher-adjacent-rescue"] == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue"
    )


def test_track2p_supported_fn_edges_filter_teacher_only_misses() -> None:
    predicted = np.asarray([[1, 2, -1]], dtype=int)
    reference = np.asarray([[1, 2, 3]], dtype=int)
    track2p = np.asarray([[1, 2, 3]], dtype=int)
    policy = np.asarray([[1, 2, -1]], dtype=int)

    assert audit._track2p_supported_fn_edges(predicted, reference, track2p, policy) == (
        (1, 2, 2, 3),
    )


def test_track2p_supported_fn_edges_excludes_policy_supported_edges() -> None:
    predicted = np.asarray([[1, 2, -1]], dtype=int)
    reference = np.asarray([[1, 2, 3]], dtype=int)
    track2p = np.asarray([[1, 2, 3]], dtype=int)
    policy = np.asarray([[1, 2, 3]], dtype=int)

    assert (
        audit._track2p_supported_fn_edges(predicted, reference, track2p, policy) == ()
    )


def test_simulate_adjacent_rescue_inserts_missing_target() -> None:
    predicted = np.asarray([[10, 11, -1]], dtype=int)
    reference_complete = audit._complete_track_counter(
        np.asarray([[10, 11, 12]], dtype=int)
    )

    simulation = audit._simulate_adjacent_rescue(
        predicted, (1, 2, 11, 12), reference_complete=reference_complete
    )

    assert simulation.applied
    assert simulation.action == "insert_target"
    np.testing.assert_array_equal(simulation.candidate, [[10, 11, 12]])


def test_simulate_adjacent_rescue_rejects_duplicate_target() -> None:
    predicted = np.asarray([[10, 11, 99]], dtype=int)

    simulation = audit._simulate_adjacent_rescue(
        predicted,
        (1, 2, 11, 12),
        reference_complete=audit._complete_track_counter(np.empty((0, 3), dtype=int)),
    )

    assert not simulation.applied
    assert simulation.would_create_duplicate_source
    assert simulation.reason == "duplicate_source_or_target"


def test_score_delta_columns_report_pairwise_effect() -> None:
    baseline = {
        "pairwise_true_positives": 1,
        "pairwise_false_positives": 0,
        "pairwise_false_negatives": 1,
        "pairwise_f1": 2 / 3,
    }
    candidate = {
        "pairwise_true_positives": 2,
        "pairwise_false_positives": 0,
        "pairwise_false_negatives": 0,
        "pairwise_f1": 1.0,
    }

    delta = audit._score_delta_columns(baseline, candidate, prefix="what_if_pairwise")

    assert delta["what_if_pairwise_tp_delta"] == 1
    assert delta["what_if_pairwise_fp_delta"] == 0
    assert delta["what_if_pairwise_fn_delta"] == -1


def test_teacher_fn_parser_defaults_to_component_cleanup_settings() -> None:
    args = audit.build_arg_parser().parse_args(
        ["--data", "track2p-root", "--output", "teacher_fn.csv"]
    )

    assert args.threshold_method == "min"
    assert args.iou_distance_threshold == 12
    assert args.cell_probability_threshold == 0.5
    assert args.split_risk_threshold == 1.5
    assert args.feature_mode == "none"


def test_teacher_adjacent_parser_defaults_to_structural_order() -> None:
    args = rescue.build_arg_parser().parse_args(["--data", "track2p-root"])

    assert args.teacher_edge_order == "structural"
    assert args.allow_completing_fragment_merges is False
    assert args.allow_completing_seed_source_backfill is False


def test_teacher_adjacent_parser_accepts_dynamic_structural_order() -> None:
    args = rescue.build_arg_parser().parse_args(
        ["--data", "track2p-root", "--teacher-edge-order", "dynamic-structural"]
    )

    assert args.teacher_edge_order == "dynamic-structural"


def test_teacher_adjacent_rescue_extends_seed_anchored_chain() -> None:
    predicted = np.asarray([[10, -1, -1, 13, -1, -1]], dtype=int)
    teacher = np.asarray([[10, -1, -1, 13, 14, 15]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0
    )

    np.testing.assert_array_equal(report.tracks, [[10, -1, -1, 13, 14, 15]])
    assert [row["applied"] for row in report.rows] == [1, 1]


def test_teacher_adjacent_rescue_rejects_complete_row_by_default() -> None:
    predicted = np.asarray([[10, 11, -1]], dtype=int)
    teacher = np.asarray([[10, 11, 12]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0
    )

    np.testing.assert_array_equal(report.tracks, predicted)
    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "would_complete_track"


def test_teacher_adjacent_rescue_can_allow_complete_row() -> None:
    predicted = np.asarray([[10, 11, -1]], dtype=int)
    teacher = np.asarray([[10, 11, 12]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0, allow_completing_rescue=True
    )

    np.testing.assert_array_equal(report.tracks, [[10, 11, 12]])
    assert report.rows[0]["applied"] == 1


def test_teacher_adjacent_rescue_backfills_missing_internal_source() -> None:
    predicted = np.asarray([[10, -1, 12, -1]], dtype=int)
    teacher = np.asarray([[-1, 11, 12, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0
    )

    np.testing.assert_array_equal(report.tracks, [[10, 11, 12, -1]])
    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_insert_source"


def test_teacher_adjacent_rescue_can_disable_source_backfill() -> None:
    predicted = np.asarray([[10, -1, 12, -1]], dtype=int)
    teacher = np.asarray([[-1, 11, 12, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0, allow_source_backfill=False
    )

    np.testing.assert_array_equal(report.tracks, predicted)
    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "missing_or_ambiguous_source"


def test_teacher_adjacent_rescue_source_insert_alias_controls_backfill() -> None:
    predicted = np.asarray([[10, -1, 12, -1]], dtype=int)
    teacher = np.asarray([[-1, 11, 12, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0, allow_source_inserts=False
    )

    np.testing.assert_array_equal(report.tracks, predicted)
    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "missing_or_ambiguous_source"


def test_teacher_adjacent_rescue_seed_backfill_is_opt_in() -> None:
    predicted = np.asarray([[-1, 11, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 11, -1, -1]], dtype=int)

    default_report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0
    )
    np.testing.assert_array_equal(default_report.tracks, predicted)
    assert default_report.rows[0]["reason"] == "target_not_seed_anchored"

    opt_in_report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0, allow_seed_source_backfill=True
    )
    np.testing.assert_array_equal(opt_in_report.tracks, [[10, 11, 12, -1]])
    assert opt_in_report.rows[0]["applied"] == 1
    assert opt_in_report.rows[0]["reason"] == "accepted_insert_source"


def test_teacher_adjacent_rescue_seed_backfill_still_rejects_complete_row() -> None:
    predicted = np.asarray([[-1, 11, 12]], dtype=int)
    teacher = np.asarray([[10, 11, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
    )

    np.testing.assert_array_equal(report.tracks, predicted)
    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "would_complete_track"


def test_teacher_adjacent_rescue_can_complete_seed_backfill_when_enabled() -> None:
    predicted = np.asarray([[-1, 11, 12]], dtype=int)
    teacher = np.asarray([[10, 11, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_seed_source_backfill=True,
        allow_completing_seed_source_backfill=True,
    )

    np.testing.assert_array_equal(report.tracks, [[10, 11, 12]])
    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_insert_source"


def test_teacher_adjacent_rescue_structural_order_prefers_source_backfill() -> None:
    predicted = np.asarray([[10, -1, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 99, -1, -1], [-1, 11, 12, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0
    )

    np.testing.assert_array_equal(report.tracks, [[10, 11, 12, -1]])
    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_insert_source"
    assert report.rows[1]["applied"] == 0
    assert report.rows[1]["reason"] == "source_has_target_conflict"


def test_teacher_adjacent_rescue_lexicographic_order_preserves_old_behavior() -> None:
    predicted = np.asarray([[10, -1, 12, -1]], dtype=int)
    teacher = np.asarray([[10, 99, -1, -1], [-1, 11, 12, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        edge_order="lexicographic",
    )

    np.testing.assert_array_equal(report.tracks, [[10, 99, 12, -1]])
    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_insert_target"
    assert report.rows[1]["applied"] == 0
    assert report.rows[1]["reason"] == "target_has_source_conflict"


def test_teacher_adjacent_rescue_dynamic_structural_recomputes_after_accept() -> None:
    predicted = np.asarray(
        [
            [10, -1, -1, -1, -1],
            [-1, -1, 12, 13, -1],
        ],
        dtype=int,
    )
    teacher = np.asarray(
        [
            [10, 11, 1, -1, -1],
            [-1, 11, 12, 13, -1],
        ],
        dtype=int,
    )

    static_report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0, edge_order="structural"
    )
    dynamic_report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0, edge_order="dynamic-structural"
    )

    np.testing.assert_array_equal(
        static_report.tracks,
        [[10, 11, 1, -1, -1], [-1, -1, 12, 13, -1]],
    )
    np.testing.assert_array_equal(dynamic_report.tracks, [[10, 11, 12, 13, -1]])
    assert any(
        row["reason"] == "accepted_merge_fragments" for row in dynamic_report.rows
    )


def test_teacher_adjacent_rescue_merges_compatible_fragments() -> None:
    predicted = np.asarray(
        [
            [10, 11, -1, -1, -1],
            [10, -1, 12, 13, -1],
        ],
        dtype=int,
    )
    teacher = np.asarray([[10, 11, 12, 13, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0
    )

    np.testing.assert_array_equal(report.tracks, [[10, 11, 12, 13, -1]])
    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_merge_fragments"


def test_teacher_adjacent_rescue_merges_fragments_from_reverse_edge_order() -> None:
    predicted = np.asarray(
        [
            [10, 11, -1, -1, -1],
            [10, -1, 12, 13, -1],
        ],
        dtype=int,
    )
    teacher = np.asarray([[10, -1, 12, 13, -1], [10, 11, 12, -1, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0
    )

    np.testing.assert_array_equal(report.tracks, [[10, 11, 12, 13, -1]])
    assert any(row["reason"] == "accepted_merge_fragments" for row in report.rows)


def test_teacher_adjacent_rescue_can_disable_fragment_merges() -> None:
    predicted = np.asarray(
        [
            [10, 11, -1, -1, -1],
            [10, -1, 12, 13, -1],
        ],
        dtype=int,
    )
    teacher = np.asarray([[10, 11, 12, 13, -1]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0, allow_fragment_merges=False
    )

    np.testing.assert_array_equal(report.tracks, predicted)
    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "target_already_claimed"


def test_teacher_adjacent_rescue_rejects_complete_fragment_merge_by_default() -> None:
    predicted = np.asarray([[10, 11, -1], [10, -1, 12]], dtype=int)
    teacher = np.asarray([[10, 11, 12]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0
    )

    np.testing.assert_array_equal(report.tracks, predicted)
    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "would_complete_track"


def test_teacher_adjacent_rescue_can_allow_complete_fragment_merge_only() -> None:
    predicted = np.asarray([[10, 11, -1], [10, -1, 12]], dtype=int)
    teacher = np.asarray([[10, 11, 12]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_completing_fragment_merges=True,
    )

    np.testing.assert_array_equal(report.tracks, [[10, 11, 12]])
    assert report.rows[0]["applied"] == 1
    assert report.rows[0]["reason"] == "accepted_merge_fragments"


def test_completing_fragment_merge_flag_does_not_allow_target_insert() -> None:
    predicted = np.asarray([[10, 11, -1]], dtype=int)
    teacher = np.asarray([[10, 11, 12]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted,
        teacher,
        seed_session=0,
        allow_completing_fragment_merges=True,
    )

    np.testing.assert_array_equal(report.tracks, predicted)
    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "would_complete_track"


def test_teacher_adjacent_rescue_rejects_seedless_partial_component() -> None:
    predicted = np.asarray([[-1, 11, -1]], dtype=int)
    teacher = np.asarray([[-1, 11, 12]], dtype=int)

    report = rescue.apply_teacher_adjacent_rescue_edges(
        predicted, teacher, seed_session=0
    )

    np.testing.assert_array_equal(report.tracks, predicted)
    assert report.rows[0]["applied"] == 0
    assert report.rows[0]["reason"] == "source_not_seed_anchored"
