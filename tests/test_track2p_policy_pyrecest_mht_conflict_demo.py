from __future__ import annotations

import pytest
from bayescatrack.experiments import track2p_policy_pyrecest_mht_conflict_demo as demo

_TRUE_COLLISION_EDGE = "demo:1:2:11:12:0"
_FALSE_COLLISION_EDGE = "demo:1:2:21:12:0"
_FALSE_CONTINUATION_EDGE = "demo:1:2:31:99:0"


def _by_arm(results):
    return {result.arm: result for result in results}


def test_mht_and_greedy_select_different_edit_sets():
    scenario = demo.build_default_scenario()
    arms = _by_arm(demo.evaluate_scenario(scenario))

    greedy = arms["greedy"]
    mht = arms["mht"]

    # Conflict-blind gating removes all three candidates, including the *true*
    # half of the collision; the joint MHT keeps the conflicting pair mutually
    # exclusive and removes only the worse collision edge plus the independent
    # false continuation.
    assert set(greedy.removed_candidate_ids) == {
        _TRUE_COLLISION_EDGE,
        _FALSE_COLLISION_EDGE,
        _FALSE_CONTINUATION_EDGE,
    }
    assert set(mht.removed_candidate_ids) == {
        _FALSE_COLLISION_EDGE,
        _FALSE_CONTINUATION_EDGE,
    }
    assert _TRUE_COLLISION_EDGE not in set(mht.removed_candidate_ids)
    assert mht.removed_candidate_ids != greedy.removed_candidate_ids


def test_only_mht_selection_is_conflict_free():
    scenario = demo.build_default_scenario()
    arms = _by_arm(demo.evaluate_scenario(scenario))

    assert arms["greedy"].conflict_free is False
    assert arms["mht"].conflict_free is True
    assert arms["deterministic"].conflict_free is True


def test_deterministic_arm_screens_out_collision_edges():
    scenario = demo.build_default_scenario()
    arms = _by_arm(demo.evaluate_scenario(scenario))

    # The rank gate only admits the non-competing false continuation; both halves
    # of the merge collision share target ROI 12 and are screened out.
    assert set(arms["deterministic"].removed_candidate_ids) == {
        _FALSE_CONTINUATION_EDGE,
    }


@pytest.mark.parametrize("max_edits", [0, -1])
def test_demo_rejects_invalid_max_edits(max_edits):
    scenario = demo.build_default_scenario()

    with pytest.raises(ValueError, match="max_edits"):
        demo.evaluate_scenario(scenario, max_edits=max_edits)

    with pytest.raises(ValueError, match="max_edits"):
        demo.greedy_select(
            scenario.candidates,
            score_threshold=0.0,
            max_edits=max_edits,
        )


def test_metric_ordering_shows_mht_repairs_what_greedy_corrupts():
    scenario = demo.build_default_scenario()
    arms = _by_arm(demo.evaluate_scenario(scenario))

    baseline = arms["baseline"]
    deterministic = arms["deterministic"]
    greedy = arms["greedy"]
    mht = arms["mht"]

    assert baseline.pairwise_f1 == pytest.approx(2 / 3, abs=1e-4)
    assert baseline.complete_track_f1 == pytest.approx(1 / 3, abs=1e-4)
    assert deterministic.pairwise_f1 == pytest.approx(8 / 11, abs=1e-4)
    assert deterministic.complete_track_f1 == pytest.approx(2 / 5, abs=1e-4)
    assert greedy.pairwise_f1 == pytest.approx(2 / 3, abs=1e-4)
    assert greedy.complete_track_f1 == pytest.approx(0.0, abs=1e-4)
    assert mht.pairwise_f1 == pytest.approx(0.8, abs=1e-4)
    assert mht.complete_track_f1 == pytest.approx(0.5, abs=1e-4)

    # Greedy collision repair destroys a true complete track (worse than doing
    # nothing); the MHT strictly improves over baseline on both metrics.
    assert greedy.complete_track_f1 < baseline.complete_track_f1
    assert mht.pairwise_f1 > baseline.pairwise_f1
    assert mht.complete_track_f1 > baseline.complete_track_f1
    assert mht.pairwise_f1 > greedy.pairwise_f1
    assert mht.complete_track_f1 > greedy.complete_track_f1


def test_selection_does_not_read_ground_truth_field():
    scenario = demo.build_default_scenario()
    stripped = demo.DemoScenario(
        name=scenario.name,
        reference=scenario.reference,
        predicted=scenario.predicted,
        candidates=tuple(
            {key: value for key, value in row.items() if key != "edge_truth"}
            for row in scenario.candidates
        ),
    )

    config = demo.ResidualMHTConfig(
        max_edits=3, max_hypotheses=64, edit_penalty=0.25, score_threshold=0.0
    )
    with_truth = demo.mht_select(scenario.candidates, config=config)
    without_truth = demo.mht_select(stripped.candidates, config=config)

    assert [demo.residual_mht._candidate_id(row) for row in with_truth] == [
        demo.residual_mht._candidate_id(row) for row in without_truth
    ]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_demo_parser_rejects_invalid_max_edits(value):
    with pytest.raises(SystemExit):
        demo.build_arg_parser().parse_args(["--max-edits", value])
