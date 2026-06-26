from __future__ import annotations

import pytest

conflict_demo = pytest.importorskip(
    "bayescatrack.experiments.track2p_policy_full_mht_conflict_demo"
)


def _by_arm(results):
    return {result.arm: result for result in results}


def test_full_mht_beam_rescues_dead_end_history():
    results = _by_arm(
        conflict_demo.evaluate_demo(
            scenario=conflict_demo.build_default_scenario(),
            mht_beam_width=2,
            scan_hypotheses=2,
        )
    )

    greedy = results["greedy"]
    full_mht = results["full_mht"]

    assert greedy.path == (1, 10, -1)
    assert full_mht.path == (1, 20, 30)
    assert full_mht.score > greedy.score
    assert full_mht.pairwise_f1 >= greedy.pairwise_f1
    assert full_mht.complete_track_f1 > greedy.complete_track_f1


def test_full_mht_improves_complete_track_when_greedy_is_pairwise_good():
    results = _by_arm(
        conflict_demo.evaluate_demo(
            scenario=conflict_demo.build_pairwise_good_complete_bad_scenario(
                stable_tracks=20
            ),
            mht_beam_width=2,
            scan_hypotheses=2,
        )
    )

    greedy = results["greedy"]
    full_mht = results["full_mht"]

    assert greedy.path == (1, 20, 31, 41)
    assert full_mht.path == (1, 20, 30, 40)
    assert greedy.pairwise_f1 > 0.9
    assert full_mht.score > greedy.score
    assert full_mht.pairwise_f1 >= greedy.pairwise_f1
    assert full_mht.complete_track_f1 > greedy.complete_track_f1
