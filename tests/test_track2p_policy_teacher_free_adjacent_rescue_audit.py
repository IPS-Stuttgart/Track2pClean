import inspect

from bayescatrack import cli
from bayescatrack.experiments.track2p_policy_teacher_free_adjacent_rescue_ranking_audit import (
    teacher_free_adjacent_ranking_score,
)


def test_teacher_free_adjacent_rescue_audit_cli_aliases() -> None:
    canonical = "track2p-policy-teacher-free-adjacent-rescue-ranking-audit"
    assert canonical in cli._BENCHMARK_COMMANDS
    assert cli._BENCHMARK_ALIASES[
        "track2p-teacher-free-adjacent-rescue-ranking-audit"
    ] == canonical
    assert cli._BENCHMARK_ALIASES[
        "track2p-component-teacher-free-adjacent-rescue-ranking-audit"
    ] == canonical


def test_teacher_free_adjacent_rank_score_has_no_audit_label_inputs() -> None:
    forbidden = {
        "track2p_supported",
        "edge_status_against_gt",
        "is_gt_suffix_path",
        "reference_track_id",
        "pairwise_tp_delta_if_added",
        "pairwise_fp_delta_if_added",
        "pairwise_fn_delta_if_added",
        "complete_tp_delta_if_added",
        "complete_fp_delta_if_added",
        "complete_fn_delta_if_added",
        "would_break_complete_tp",
        "would_create_complete_fp",
    }
    parameters = set(inspect.signature(teacher_free_adjacent_ranking_score).parameters)

    assert parameters.isdisjoint(forbidden)
    assert teacher_free_adjacent_ranking_score(
        registered_iou=0.4,
        shifted_iou=0.5,
        roi_aware_score=0.35,
        centroid_distance=4.0,
        area_ratio=0.9,
        cell_probability_a=0.8,
        cell_probability_b=0.9,
        row_rank=1,
        column_rank=2,
        row_margin=0.1,
        column_margin=0.2,
        threshold_margin=0.05,
        activity_similarity=0.5,
        growth_residual_mahalanobis=3.0,
        two_edge_motion_consistency=0.7,
        would_complete_predicted_row=0,
        would_merge_components=1,
    ) > 0.0
