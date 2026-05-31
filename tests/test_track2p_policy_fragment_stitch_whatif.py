import numpy as np
from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_fragment_stitch_whatif as audit


def test_fragment_stitch_whatif_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-fragment-stitch-whatif"]

    assert canonical == "track2p-policy-fragment-stitch-whatif"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-fragment-stitch-whatif"] == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_fragment_stitch_whatif"
    )


def test_minimal_repair_plan_prefers_fragment_merge() -> None:
    predicted = np.asarray(
        [
            [10, 11, -1, -1],
            [-1, -1, 12, 13],
            [10, 99, 98, 97],
        ],
        dtype=int,
    )
    track = (10, 11, 12, 13)

    plan = audit._minimal_repair_plan(predicted, track)

    assert plan.selected_rows == (0, 1)
    assert plan.sessions_present == (0, 1, 2, 3)
    assert plan.component_merges == 1
    assert plan.edge_additions == 1
    assert plan.edge_swaps == 0
    assert audit._edge_list(plan.missing_edges) == "1:11->2:12"


def test_apply_repair_plan_replaces_fragments_with_reference_track() -> None:
    predicted = np.asarray([[10, 11, -1], [-1, 11, 12], [7, 8, 9]], dtype=int)
    track = (10, 11, 12)
    plan = audit._minimal_repair_plan(predicted, track)

    candidate = audit._apply_repair_plan(predicted, track, plan)

    assert candidate.tolist() == [[10, 11, 12], [7, 8, 9]]


def test_duplicate_flags_detect_same_source_and_target_conflicts() -> None:
    predicted = np.asarray(
        [
            [1, -1, -1],
            [-1, 20, -1],
            [1, 99, -1],
            [-1, 88, 30],
        ],
        dtype=int,
    )
    track = (1, 20, 30)
    plan = audit._minimal_repair_plan(predicted, track)

    duplicate_source, duplicate_target = audit._duplicate_flags(
        predicted, track, plan.selected_rows
    )

    assert duplicate_source
    assert duplicate_target


def test_score_delta_is_candidate_minus_baseline() -> None:
    baseline = {
        "pairwise_true_positives": 2,
        "pairwise_false_positives": 3,
        "pairwise_false_negatives": 4,
        "complete_track_true_positives": 5,
        "complete_track_false_positives": 6,
        "complete_track_false_negatives": 7,
    }
    candidate = {
        "pairwise_true_positives": 4,
        "pairwise_false_positives": 1,
        "pairwise_false_negatives": 3,
        "complete_track_true_positives": 6,
        "complete_track_false_positives": 6,
        "complete_track_false_negatives": 5,
    }

    delta = audit._score_delta(baseline, candidate)

    assert delta["pairwise_true_positives"] == 2
    assert delta["pairwise_false_positives"] == -2
    assert delta["pairwise_false_negatives"] == -1
    assert delta["complete_track_true_positives"] == 1
    assert delta["complete_track_false_negatives"] == -2


def test_global_counts_accepts_generator_input() -> None:
    rows = (
        {
            "pairwise_true_positives": index,
            "pairwise_false_positives": 1,
            "pairwise_false_negatives": 2,
            "complete_track_true_positives": 3,
            "complete_track_false_positives": 4,
            "complete_track_false_negatives": 5,
        }
        for index in (1, 2)
    )

    counts = audit._global_counts(rows)

    assert counts["pairwise_true_positives"] == 3
    assert counts["pairwise_false_positives"] == 2
    assert counts["pairwise_false_negatives"] == 4
    assert counts["complete_track_true_positives"] == 6
    assert counts["complete_track_false_positives"] == 8
    assert counts["complete_track_false_negatives"] == 10
