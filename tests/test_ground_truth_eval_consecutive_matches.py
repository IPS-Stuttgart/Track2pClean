from __future__ import annotations

from bayescatrack.ground_truth_eval import tracks_from_consecutive_matches


def test_tracks_from_consecutive_matches_infers_seed_rois_from_first_match():
    table = tracks_from_consecutive_matches(
        ("day0", "day1", "day2"),
        [
            {5: 10, 2: 20},
            {10: 100, 20: 200},
        ],
    )

    assert table.session_names == ("day0", "day1", "day2")
    assert table.tracks.tolist() == [
        [2, 20, 200],
        [5, 10, 100],
    ]


def test_tracks_from_consecutive_matches_keeps_explicit_unmatched_seed_rois():
    table = tracks_from_consecutive_matches(
        ("day0", "day1"),
        [{1: 11}],
        start_roi_indices=[0, 1],
    )

    assert table.tracks.tolist() == [
        [0, -1],
        [1, 11],
    ]
