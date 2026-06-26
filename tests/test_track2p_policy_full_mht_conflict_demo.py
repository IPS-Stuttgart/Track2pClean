from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("pyrecest")

from bayescatrack.experiments import (  # noqa: E402
    track2p_policy_full_mht_conflict_demo as demo,
)


def test_full_mht_conflict_demo_mht_history_beats_greedy():
    results = {result.arm: result for result in demo.evaluate_demo()}

    assert results["greedy"].path == (1, 10, -1)
    assert results["full_mht"].path == (1, 20, 30)
    assert results["full_mht"].score > results["greedy"].score
    assert results["full_mht"].complete_track_f1 > results["greedy"].complete_track_f1


def test_full_mht_conflict_demo_pairwise_good_can_be_complete_bad():
    scenario = demo.build_pairwise_good_complete_bad_scenario(stable_tracks=20)
    results = {result.arm: result for result in demo.evaluate_demo(scenario=scenario)}

    assert results["greedy"].path == (1, 20, 31, 41)
    assert results["full_mht"].path == (1, 20, 30, 40)
    assert results["greedy"].pairwise_f1 > 0.95
    assert results["greedy"].pairwise_f1 > results["greedy"].complete_track_f1
    assert results["full_mht"].complete_track_f1 > results["greedy"].complete_track_f1
    assert results["full_mht"].pairwise_f1 >= results["greedy"].pairwise_f1


def test_full_mht_conflict_demo_selection_is_reference_independent():
    scenario = demo.build_pairwise_good_complete_bad_scenario(stable_tracks=8)
    reference_favoring_greedy = np.asarray(scenario.reference, dtype=int).copy()
    reference_favoring_greedy[0] = np.asarray([1, 20, 31, 41], dtype=int)
    adversarial = demo.DemoScenario(
        name="reference-favors-greedy",
        reference=reference_favoring_greedy,
        seed_track=scenario.seed_track,
        edge_scores=scenario.edge_scores,
    )

    original = {result.arm: result for result in demo.evaluate_demo(scenario=scenario)}
    adversarial_results = {
        result.arm: result for result in demo.evaluate_demo(scenario=adversarial)
    }

    assert original["greedy"].path == adversarial_results["greedy"].path
    assert original["full_mht"].path == adversarial_results["full_mht"].path
    assert adversarial_results["greedy"].complete_track_f1 > original["greedy"].complete_track_f1
    assert adversarial_results["full_mht"].path == (1, 20, 30, 40)


def test_full_mht_conflict_demo_csv_rows_are_stable():
    rows = demo._result_rows(demo.evaluate_demo())

    assert [row["arm"] for row in rows] == ["greedy", "full_mht"]
    assert rows[0]["path"] == "1->10->-1"
    assert rows[1]["path"] == "1->20->30"
