from __future__ import annotations

import numpy as np
from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
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
