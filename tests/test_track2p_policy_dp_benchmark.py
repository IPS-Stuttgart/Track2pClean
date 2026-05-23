from bayescatrack.experiments.track2p_policy_dp_benchmark import (
    PolicyCandidate,
    Track2pPolicyDPConfig,
    TrackPath,
    _best_track_path,
    _best_track_paths,
    select_non_conflicting_paths,
)


def _candidate(source_session, target_session, source_roi, target_roi, score):
    return PolicyCandidate(
        source_session=source_session,
        target_session=target_session,
        source_roi=source_roi,
        target_roi=target_roi,
        score=score,
        iou=0.5,
        threshold=0.25,
        accepted_by_threshold=True,
        selected_by_hungarian=True,
    )


def test_dp_prefers_rescue_edge_with_downstream_support() -> None:
    config = Track2pPolicyDPConfig(beam_width=4)
    candidates_by_source = {
        (0, 0): (
            _candidate(0, 1, 0, 1, 1.0),
            _candidate(0, 1, 0, 2, 0.8),
        ),
        (1, 2): (_candidate(1, 2, 2, 3, 2.0),),
    }

    path = _best_track_path(
        start_roi=0,
        n_sessions=3,
        candidates_by_source=candidates_by_source,
        config=config,
    )

    assert path is not None
    assert path.row == (0, 2, 3)


def test_dp_can_return_alternate_paths_for_global_selection() -> None:
    config = Track2pPolicyDPConfig(beam_width=4)
    candidates_by_source = {
        (0, 0): (
            _candidate(0, 1, 0, 1, 10.0),
            _candidate(0, 1, 0, 2, 8.0),
        ),
    }

    paths = _best_track_paths(
        start_roi=0,
        n_sessions=2,
        candidates_by_source=candidates_by_source,
        config=config,
        max_paths=2,
    )

    assert [path.row for path in paths] == [
        (0, 1),
        (0, 2),
    ]
    assert [path.score for path in paths] == [10.0, 8.0]


def test_dp_supports_one_gap_repair_candidate() -> None:
    config = Track2pPolicyDPConfig(beam_width=4, max_gap=2)
    candidates_by_source = {
        (0, 0): (_candidate(0, 2, 0, 5, 1.5),),
    }

    path = _best_track_path(
        start_roi=0,
        n_sessions=3,
        candidates_by_source=candidates_by_source,
        config=config,
    )

    assert path is not None
    assert path.row == (0, -1, 5)


def test_select_non_conflicting_paths_keeps_highest_scoring_track() -> None:
    selected = select_non_conflicting_paths(
        (
            TrackPath(row=(0, 1, 2), score=10.0),
            TrackPath(row=(3, 1, 4), score=20.0),
            TrackPath(row=(5, 6, 7), score=1.0),
        )
    )

    assert [path.row for path in selected] == [(3, 1, 4), (5, 6, 7)]


def test_select_non_conflicting_paths_finds_component_optimum() -> None:
    selected = select_non_conflicting_paths(
        (
            TrackPath(row=(0, 1, -1), score=10.0),
            TrackPath(row=(0, 2, -1), score=6.0),
            TrackPath(row=(3, 1, -1), score=6.0),
            TrackPath(row=(4, 5, -1), score=-2.0),
        )
    )

    assert [path.row for path in selected] == [
        (0, 2, -1),
        (3, 1, -1),
        (4, 5, -1),
    ]
