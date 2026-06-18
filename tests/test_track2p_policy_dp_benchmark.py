import pytest
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


@pytest.mark.parametrize(
    "field",
    [
        "row_top_k",
        "beam_width",
        "max_gap",
        "path_candidates_per_seed",
        "path_selection_beam_width",
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, "2", float("nan"), float("inf")],
)
def test_dp_config_rejects_invalid_positive_integer_controls(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        Track2pPolicyDPConfig(**{field: value})


@pytest.mark.parametrize("fill_value", [True, False, 1.5, "2", float("nan")])
def test_dp_config_rejects_invalid_fill_value(fill_value: object) -> None:
    with pytest.raises(ValueError, match="fill_value"):
        Track2pPolicyDPConfig(fill_value=fill_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("iou_distance_threshold", True),
        ("iou_distance_threshold", False),
        ("iou_distance_threshold", float("nan")),
        ("iou_distance_threshold", float("inf")),
        ("iou_distance_threshold", -0.1),
        ("rescue_min_iou", True),
        ("rescue_min_iou", False),
        ("rescue_min_iou", float("nan")),
        ("rescue_min_iou", float("inf")),
        ("rescue_min_iou", -0.1),
        ("rescue_min_iou", 1.1),
        ("threshold_rescue_margin", True),
        ("threshold_rescue_margin", False),
        ("threshold_rescue_margin", float("nan")),
        ("threshold_rescue_margin", float("inf")),
        ("threshold_rescue_margin", -0.1),
        ("accepted_bonus", True),
        ("accepted_bonus", False),
        ("accepted_bonus", float("nan")),
        ("accepted_bonus", float("inf")),
        ("rescue_penalty", True),
        ("rescue_penalty", False),
        ("rescue_penalty", float("nan")),
        ("rescue_penalty", float("inf")),
        ("gap_penalty", True),
        ("gap_penalty", False),
        ("gap_penalty", float("nan")),
        ("gap_penalty", float("inf")),
        ("threshold_margin_weight", True),
        ("threshold_margin_weight", False),
        ("threshold_margin_weight", float("nan")),
        ("threshold_margin_weight", float("inf")),
        ("logit_epsilon", True),
        ("logit_epsilon", False),
        ("logit_epsilon", float("nan")),
        ("logit_epsilon", float("inf")),
        ("logit_epsilon", 0.0),
        ("logit_epsilon", 0.5),
    ],
)
def test_dp_config_rejects_invalid_float_controls(
    field: str, value: float | bool
) -> None:
    with pytest.raises(ValueError, match=field):
        Track2pPolicyDPConfig(**{field: value})
