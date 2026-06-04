import numpy as np
from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_coherence_pareto_whatif as audit
from pyrecest.utils.track_edit_whatif import TrackEdit, score_track_edit_delta


def test_coherence_pareto_whatif_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-coherence-pareto-whatif"]

    assert canonical == "track2p-policy-coherence-pareto-whatif"
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_coherence_pareto_whatif"
    )


def test_remove_edge_splits_component_and_scores_exact_delta() -> None:
    predicted = np.asarray([[1, 2, 99, 4]], dtype=int)
    reference = np.asarray([[1, 2, 3, 4]], dtype=int)

    simulation = audit._simulate_remove_edge(predicted, (1, 2, 2, 99))
    delta = score_track_edit_delta(
        predicted,
        reference,
        TrackEdit(
            kind="remove_link",
            session_a=1,
            session_b=2,
            source_observation=2,
            target_observation=99,
        ),
        count_duplicates=True,
    )

    assert simulation.applied
    assert simulation.candidate.tolist() == [[1, 2, -1, -1], [-1, -1, 99, 4]]
    assert delta.pairwise_fp_delta == -1
    assert delta.complete_fp_delta == -1


def test_add_edge_inserts_clean_missing_target() -> None:
    predicted = np.asarray([[1, 2, -1]], dtype=int)
    reference = np.asarray([[1, 2, 3]], dtype=int)

    simulation = audit._simulate_add_edge(predicted, (1, 2, 2, 3))
    delta = score_track_edit_delta(
        predicted,
        reference,
        TrackEdit(
            kind="add_link",
            session_a=1,
            session_b=2,
            source_observation=2,
            target_observation=3,
        ),
        count_duplicates=True,
    )

    assert simulation.applied
    assert simulation.action == "insert_target"
    assert simulation.candidate.tolist() == [[1, 2, 3]]
    assert delta.pairwise_tp_delta == 1
    assert delta.pairwise_fn_delta == -1
    assert delta.complete_tp_delta == 1


def test_add_edge_rejects_conflict_for_swap_path() -> None:
    predicted = np.asarray([[1, 2, 99]], dtype=int)

    simulation = audit._simulate_add_edge(predicted, (1, 2, 2, 3))

    assert not simulation.applied
    assert simulation.duplicate_source
    assert simulation.reason == "duplicate_source_or_target"


def test_swap_conflict_coupled_edge_replaces_wrong_adjacent_link() -> None:
    predicted = np.asarray([[1, 2, 99]], dtype=int)
    reference = np.asarray([[1, 2, 3]], dtype=int)

    simulation = audit._simulate_swap_edge(
        predicted,
        missing_edge=(1, 2, 2, 3),
        wrong_edge=(1, 2, 2, 99),
    )
    delta = score_track_edit_delta(
        predicted,
        reference,
        TrackEdit(
            kind="swap_link",
            session_a=1,
            session_b=2,
            source_observation=2,
            target_observation=3,
            metadata={
                "remove_session_a": 1,
                "remove_session_b": 2,
                "remove_source_observation": 2,
                "remove_target_observation": 99,
            },
        ),
        count_duplicates=True,
    )

    assert simulation.applied
    assert simulation.candidate.tolist() == [[1, 2, 3]]
    assert delta.pairwise_tp_delta == 1
    assert delta.pairwise_fp_delta == -1
    assert delta.pairwise_fn_delta == -1
    assert delta.complete_tp_delta == 1
    assert delta.complete_fp_delta == -1


def test_sum_scores_accepts_single_pass_iterables() -> None:
    rows = (
        {
            "pairwise_true_positives": value,
            "pairwise_false_positives": 1,
            "pairwise_false_negatives": 2,
            "complete_track_true_positives": 3,
            "complete_track_false_positives": 4,
            "complete_track_false_negatives": 5,
        }
        for value in (10, 20)
    )

    assert audit._sum_scores(rows) == {
        "pairwise_true_positives": 30,
        "pairwise_false_positives": 2,
        "pairwise_false_negatives": 4,
        "complete_track_true_positives": 6,
        "complete_track_false_positives": 8,
        "complete_track_false_negatives": 10,
    }
