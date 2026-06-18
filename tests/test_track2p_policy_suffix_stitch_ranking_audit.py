import numpy as np
import pytest
from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_suffix_stitch_ranking_audit as audit


def test_suffix_stitch_ranking_audit_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-suffix-stitch-ranking-audit"]

    assert canonical == "track2p-policy-suffix-stitch-ranking-audit"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-suffix-stitch-ranking-audit"]
        == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_suffix_stitch_ranking_audit"
    )


@pytest.mark.parametrize(
    "option",
    ["--max-suffix-length", "--edge-top-k", "--path-beam-width"],
)
def test_suffix_stitch_ranking_parser_rejects_nonpositive_search_budgets(
    option: str,
) -> None:
    with pytest.raises(SystemExit):
        audit.build_arg_parser().parse_args(
            [
                "--data",
                "track2p-root",
                "--output",
                "ranking.csv",
                option,
                "0",
            ]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_suffix_length", 0),
        ("edge_top_k", 0),
        ("path_beam_width", 0),
    ],
)
def test_ranked_suffix_paths_rejects_nonpositive_search_budgets(
    field: str,
    value: int,
) -> None:
    kwargs = {
        "max_suffix_length": 2,
        "edge_top_k": 5,
        "path_beam_width": 5,
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        audit._ranked_suffix_paths(
            np.asarray([[1, 2, -1]], dtype=int),
            np.asarray([[1, 2, 3]], dtype=int),
            subject="s",
            feature_cache=None,  # type: ignore[arg-type]
            **kwargs,
        )


def test_suffix_fragment_span_requires_contiguous_observations() -> None:
    assert audit._suffix_fragment_span(np.asarray([1, 2, 3, -1])) == (0, 2)
    assert audit._suffix_fragment_span(np.asarray([-1, 2, 3, -1])) == (1, 2)
    assert audit._suffix_fragment_span(np.asarray([1, -1, 3, -1])) is None
    assert audit._suffix_fragment_span(np.asarray([-1, -1, -1])) is None


def test_ranked_suffix_paths_skips_fragments_beyond_max_suffix_length() -> None:
    predicted = np.asarray([[1, 2, -1, -1, -1]], dtype=int)
    reference = np.asarray([[1, 2, 3, 4, 5]], dtype=int)

    paths = audit._ranked_suffix_paths(
        predicted,
        reference,
        subject="s",
        feature_cache=None,  # type: ignore[arg-type]
        max_suffix_length=2,
        edge_top_k=5,
        path_beam_width=5,
    )

    assert paths == ()


def test_expand_paths_for_fragment_uses_local_beam_completion(monkeypatch) -> None:
    first = audit._EdgeCandidate(
        edge=(1, 2, 20, 30),
        registered_iou=0.8,
        shifted_iou=0.7,
        roi_aware_score=0.6,
        centroid_distance=2.0,
        area_ratio=0.9,
        cell_probability_a=1.0,
        cell_probability_b=1.0,
        row_rank=1,
        column_rank=1,
        row_margin=0.2,
        column_margin=0.2,
        threshold_margin=0.3,
        activity_similarity=0.5,
        edge_score=1.0,
    )
    second = audit._EdgeCandidate(
        edge=(2, 3, 30, 40),
        registered_iou=0.7,
        shifted_iou=0.6,
        roi_aware_score=0.5,
        centroid_distance=2.0,
        area_ratio=0.9,
        cell_probability_a=1.0,
        cell_probability_b=1.0,
        row_rank=1,
        column_rank=1,
        row_margin=0.2,
        column_margin=0.2,
        threshold_margin=0.3,
        activity_similarity=0.5,
        edge_score=0.8,
    )

    def fake_top_edge_candidates(_cache, session_index, source_roi, *, top_k):
        del top_k
        if (session_index, source_roi) == (1, 20):
            return (first,)
        if (session_index, source_roi) == (2, 30):
            return (second,)
        return ()

    monkeypatch.setattr(audit, "_top_edge_candidates", fake_top_edge_candidates)
    paths = audit._expand_paths_for_fragment(
        component_id=7,
        row=(10, 20, -1, -1),
        fragment_span="0-1",
        tail_session=1,
        feature_cache=object(),  # type: ignore[arg-type]
        max_steps=2,
        edge_top_k=5,
        path_beam_width=5,
    )

    assert [tuple(edge.edge for edge in path.edges) for path in paths] == [
        ((1, 2, 20, 30),),
        ((1, 2, 20, 30), (2, 3, 30, 40)),
    ]


def test_rank_paths_orders_by_non_gt_score_before_label() -> None:
    high = audit._PathCandidate(
        component_id=0,
        fragment_row=(1, 2, -1),
        fragment_span="0-1",
        edges=(),
        path_score=0.9,
        is_gt_suffix_path=0,
    )
    gt = audit._PathCandidate(
        component_id=0,
        fragment_row=(1, 2, -1),
        fragment_span="0-1",
        edges=(),
        path_score=0.7,
        is_gt_suffix_path=1,
    )

    ranked = audit._rank_paths((gt, high))

    assert ranked[0].is_gt_suffix_path == 0
    assert ranked[0].path_rank == 1
    assert ranked[1].is_gt_suffix_path == 1
    assert ranked[1].path_rank == 2


def test_label_gt_path_matches_reference_suffix() -> None:
    edge = audit._EdgeCandidate(
        edge=(1, 2, 20, 30),
        registered_iou=0.8,
        shifted_iou=0.9,
        roi_aware_score=0.7,
        centroid_distance=2.0,
        area_ratio=0.9,
        cell_probability_a=1.0,
        cell_probability_b=1.0,
        row_rank=1,
        column_rank=1,
        row_margin=0.2,
        column_margin=0.3,
        threshold_margin=0.4,
        activity_similarity=0.5,
        edge_score=1.0,
    )
    path = audit._PathCandidate(
        component_id=0,
        fragment_row=(10, 20, -1),
        fragment_span="0-1",
        edges=(edge,),
        path_score=1.0,
    )
    fragment = np.asarray([10, 20, -1], dtype=int)
    reference = np.asarray([[10, 20, 30], [10, 20, 99]], dtype=int)

    labeled = audit._label_gt_path(path, fragment, reference)

    assert labeled.is_gt_suffix_path == 1


def test_label_gt_path_requires_reaching_final_session() -> None:
    edge = audit._EdgeCandidate(
        edge=(1, 2, 20, 30),
        registered_iou=0.8,
        shifted_iou=0.9,
        roi_aware_score=0.7,
        centroid_distance=2.0,
        area_ratio=0.9,
        cell_probability_a=1.0,
        cell_probability_b=1.0,
        row_rank=1,
        column_rank=1,
        row_margin=0.2,
        column_margin=0.3,
        threshold_margin=0.4,
        activity_similarity=0.5,
        edge_score=1.0,
    )
    path = audit._PathCandidate(
        component_id=0,
        fragment_row=(10, 20, -1, -1),
        fragment_span="0-1",
        edges=(edge,),
        path_score=1.0,
    )
    fragment = np.asarray([10, 20, -1, -1], dtype=int)
    reference = np.asarray([[10, 20, 30, 40]], dtype=int)

    labeled = audit._label_gt_path(path, fragment, reference)

    assert labeled.is_gt_suffix_path == 0


def test_summary_reports_top3_recovery_and_same_gate_count() -> None:
    rows = [
        {
            "subject": "s",
            "component_id": 1,
            "is_gt_suffix_path": 0,
            "path_rank": 1,
            "path_score": 0.8,
        },
        {
            "subject": "s",
            "component_id": 1,
            "is_gt_suffix_path": 1,
            "path_rank": 2,
            "path_score": 0.7,
        },
    ]

    summary = audit._summary_row("s", rows)

    assert summary["suffix_fragment_candidates"] == 1
    assert summary["number_of_gt_suffix_paths"] == 1
    assert summary["best_gt_suffix_path_rank"] == 2
    assert summary["top1_recovery_rate"] == 0.0
    assert summary["top3_recovery_rate"] == 1.0
    assert summary["non_gt_paths_that_would_pass_same_gate"] == 1
