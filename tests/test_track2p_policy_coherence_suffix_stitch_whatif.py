import inspect
from dataclasses import replace

import numpy as np
import pytest
from bayescatrack import cli
from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_exposure_audit as exposure,
)
from bayescatrack.experiments import track2p_policy_coherence_suffix_stitch as method
from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_stitch_whatif as audit,
)
from bayescatrack.experiments.track2p_policy_suffix_stitch_ranking_audit import (
    _EdgeCandidate,
    _PathCandidate,
)


def _edge(
    session_a: int,
    roi_a: int,
    roi_b: int,
    *,
    centroid_distance: float = 4.0,
    area_ratio: float = 0.9,
    shifted_iou: float = 0.5,
) -> _EdgeCandidate:
    return _EdgeCandidate(
        edge=(session_a, session_a + 1, roi_a, roi_b),
        registered_iou=0.2,
        shifted_iou=shifted_iou,
        roi_aware_score=0.18,
        centroid_distance=centroid_distance,
        area_ratio=area_ratio,
        cell_probability_a=0.95,
        cell_probability_b=0.90,
        row_rank=2,
        column_rank=2,
        row_margin=-0.1,
        column_margin=-0.1,
        threshold_margin=0.0,
        activity_similarity=0.0,
        edge_score=0.5,
    )


def test_coherence_suffix_stitch_whatif_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-coherence-suffix-stitch-whatif"]

    assert canonical == "track2p-policy-coherence-suffix-stitch-whatif"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-coherence-suffix-stitch-whatif"]
        == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_coherence_suffix_stitch_whatif"
    )


def test_coherence_suffix_stitch_method_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-coherence-suffix-stitch"]

    assert canonical == "track2p-policy-coherence-suffix-stitch"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-coherence-suffix-stitch"] == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_coherence_suffix_stitch"
    )


def test_coherence_suffix_exposure_audit_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-coherence-suffix-exposure-audit"]

    assert canonical == "track2p-policy-coherence-suffix-exposure-audit"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-coherence-suffix-exposure-audit"]
        == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_coherence_suffix_exposure_audit"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("suffix_path_length", 0),
        ("max_stitches_per_subject", 0),
    ],
)
def test_coherence_suffix_gate_rejects_nonpositive_integer_controls(
    field: str, value: int
) -> None:
    with pytest.raises(ValueError, match=field):
        audit.CoherenceSuffixStitchGate(**{field: value})


@pytest.mark.parametrize(
    "option",
    [
        "--suffix-path-length",
        "--max-stitches-per-subject",
        "--edge-top-k",
        "--path-beam-width",
    ],
)
def test_coherence_suffix_cli_rejects_nonpositive_integer_controls(
    option: str,
) -> None:
    with pytest.raises(SystemExit):
        audit.build_arg_parser().parse_args(
            [
                "--data",
                "track2p-root",
                "--output",
                "suffix.csv",
                option,
                "0",
            ]
        )


def test_candidate_output_is_optional() -> None:
    parser = audit.build_arg_parser()

    args = parser.parse_args(
        [
            "--data",
            "data",
            "--reference",
            "ref",
            "--output",
            "out.csv",
        ]
    )

    assert args.candidate_output is None
    assert args.aggregate_row


def test_method_output_is_subject_level_by_default() -> None:
    parser = method.build_arg_parser()

    args = parser.parse_args(
        [
            "--data",
            "data",
            "--reference",
            "ref",
            "--output",
            "out.csv",
        ]
    )

    assert args.candidate_output is None
    assert not args.aggregate_row


def test_exposure_audit_does_not_require_reference_args() -> None:
    parser = exposure.build_arg_parser()

    args = parser.parse_args(
        [
            "--data",
            "data",
            "--output",
            "out.csv",
        ]
    )

    assert not hasattr(args, "reference")
    assert args.aggregate_row


def test_coherence_gate_accepts_two_edge_final_suffix() -> None:
    predicted = np.asarray([[1, 2, 3, -1, -1]], dtype=int)
    path = _PathCandidate(
        component_id=0,
        fragment_row=(1, 2, 3, -1, -1),
        fragment_span="0-2",
        edges=(_edge(2, 3, 4), _edge(3, 4, 5)),
        path_score=0.5,
        path_rank=3,
        is_gt_suffix_path=1,
    )

    assert audit._passes_coherence_gate(
        path, predicted, gate=audit.CoherenceSuffixStitchGate()
    )


def test_coherence_gate_rejects_occupied_continuation() -> None:
    predicted = np.asarray([[1, 2, 3, 99, -1]], dtype=int)
    path = _PathCandidate(
        component_id=0,
        fragment_row=(1, 2, 3, 99, -1),
        fragment_span="0-2",
        edges=(_edge(2, 3, 4), _edge(3, 4, 5)),
        path_score=0.5,
    )

    assert not audit._passes_coherence_gate(
        path, predicted, gate=audit.CoherenceSuffixStitchGate()
    )


def test_apply_suffix_paths_fills_empty_suffix_slots() -> None:
    predicted = np.asarray([[1, 2, 3, -1, -1], [9, 8, 7, -1, -1]], dtype=int)
    path = _PathCandidate(
        component_id=0,
        fragment_row=(1, 2, 3, -1, -1),
        fragment_span="0-2",
        edges=(_edge(2, 3, 4), _edge(3, 4, 5)),
        path_score=0.5,
    )

    stitched = audit._apply_suffix_paths(predicted, (path,))

    assert stitched.tolist() == [[1, 2, 3, 4, 5], [9, 8, 7, -1, -1]]


def test_select_paths_uses_coherence_sort_and_limit() -> None:
    predicted = np.asarray([[1, 2, 3, -1, -1], [10, 20, 30, -1, -1]], dtype=int)
    weaker = _PathCandidate(
        component_id=0,
        fragment_row=(1, 2, 3, -1, -1),
        fragment_span="0-2",
        edges=(
            _edge(2, 3, 4, centroid_distance=4.8),
            _edge(3, 4, 5, centroid_distance=4.2),
        ),
        path_score=10.0,
        path_rank=1,
    )
    stronger = _PathCandidate(
        component_id=1,
        fragment_row=(10, 20, 30, -1, -1),
        fragment_span="0-2",
        edges=(
            _edge(2, 30, 40, centroid_distance=4.0),
            _edge(3, 40, 50, centroid_distance=4.0),
        ),
        path_score=1.0,
        path_rank=2,
    )

    selected = audit._select_paths(
        (weaker, stronger),
        predicted,
        gate=audit.CoherenceSuffixStitchGate(max_stitches_per_subject=1),
    )

    assert selected == (stronger,)


def test_select_paths_skips_conflicting_targets_when_limit_allows_more() -> None:
    predicted = np.asarray(
        [[1, 2, 3, -1, -1], [10, 20, 30, -1, -1], [100, 200, 300, -1, -1]],
        dtype=int,
    )
    strongest = _PathCandidate(
        component_id=0,
        fragment_row=(1, 2, 3, -1, -1),
        fragment_span="0-2",
        edges=(
            _edge(2, 3, 4, centroid_distance=2.0),
            _edge(3, 4, 5, centroid_distance=2.0),
        ),
        path_score=3.0,
        path_rank=1,
    )
    conflicting_target = _PathCandidate(
        component_id=1,
        fragment_row=(10, 20, 30, -1, -1),
        fragment_span="0-2",
        edges=(
            _edge(2, 30, 4, centroid_distance=2.1),
            _edge(3, 4, 6, centroid_distance=2.1),
        ),
        path_score=2.0,
        path_rank=2,
    )
    compatible = _PathCandidate(
        component_id=2,
        fragment_row=(100, 200, 300, -1, -1),
        fragment_span="0-2",
        edges=(
            _edge(2, 300, 400, centroid_distance=3.0),
            _edge(3, 400, 500, centroid_distance=3.0),
        ),
        path_score=1.0,
        path_rank=3,
    )

    selected = audit._select_paths(
        (strongest, conflicting_target, compatible),
        predicted,
        gate=audit.CoherenceSuffixStitchGate(max_stitches_per_subject=2),
    )

    assert selected == (strongest, compatible)


def test_select_paths_is_invariant_to_gt_suffix_labels() -> None:
    predicted = np.asarray([[1, 2, 3, -1, -1], [10, 20, 30, -1, -1]], dtype=int)
    weaker = _PathCandidate(
        component_id=0,
        fragment_row=(1, 2, 3, -1, -1),
        fragment_span="0-2",
        edges=(
            _edge(2, 3, 4, centroid_distance=4.8),
            _edge(3, 4, 5, centroid_distance=4.2),
        ),
        path_score=10.0,
        path_rank=1,
        is_gt_suffix_path=1,
    )
    stronger = _PathCandidate(
        component_id=1,
        fragment_row=(10, 20, 30, -1, -1),
        fragment_span="0-2",
        edges=(
            _edge(2, 30, 40, centroid_distance=4.0),
            _edge(3, 40, 50, centroid_distance=4.0),
        ),
        path_score=1.0,
        path_rank=2,
        is_gt_suffix_path=0,
    )

    selected_with_gt_labels = audit._select_paths(
        (weaker, stronger),
        predicted,
        gate=audit.CoherenceSuffixStitchGate(max_stitches_per_subject=1),
    )
    selected_without_gt_labels = audit._select_paths(
        (
            replace(weaker, is_gt_suffix_path=0),
            replace(stronger, is_gt_suffix_path=0),
        ),
        predicted,
        gate=audit.CoherenceSuffixStitchGate(max_stitches_per_subject=1),
    )

    assert _selected_path_ids(selected_with_gt_labels) == _selected_path_ids(
        selected_without_gt_labels
    )
    assert _selected_path_ids(selected_with_gt_labels) == (
        (1, ((2, 3, 30, 40), (3, 4, 40, 50))),
    )


def test_selector_does_not_read_audit_rows_or_gt_labels(monkeypatch) -> None:
    predicted = np.asarray([[1, 2, 3, -1, -1]], dtype=int)
    path = _PathCandidate(
        component_id=0,
        fragment_row=(1, 2, 3, -1, -1),
        fragment_span="0-2",
        edges=(_edge(2, 3, 4), _edge(3, 4, 5)),
        path_score=0.5,
        path_rank=1,
        is_gt_suffix_path=1,
    )

    def fail_if_audit_row_is_read(*_args, **_kwargs):
        raise AssertionError("selection must not read GT-audited path rows")

    monkeypatch.setattr(audit, "_path_row", fail_if_audit_row_is_read)

    selected = audit._select_paths(
        (path,),
        predicted,
        gate=audit.CoherenceSuffixStitchGate(max_stitches_per_subject=1),
    )

    assert selected == (path,)


def test_selector_source_excludes_gt_audit_fields() -> None:
    selector_source = "\n".join(
        inspect.getsource(function)
        for function in (
            audit._select_paths,
            audit._passes_coherence_gate,
            audit._compatible_with_selected_paths,
            audit._coherence_sort_key,
            audit._path_metrics,
            audit._would_merge_complete_prediction,
        )
    )

    forbidden_tokens = (
        "is_gt_suffix_path",
        "edge_status_against_gt",
        "reference_track_id",
        "complete_tp_delta",
        "complete_fp_delta",
        "complete_fn_delta",
        "pairwise_tp_delta",
        "pairwise_fp_delta",
        "pairwise_fn_delta",
        "manual_gt",
        "reference",
        "_path_row",
        "score_track_matrices",
    )
    for token in forbidden_tokens:
        assert token not in selector_source


def test_exposure_row_reports_gate_frequency_only() -> None:
    selected = _PathCandidate(
        component_id=1,
        fragment_row=(10, 20, 30, -1, -1),
        fragment_span="0-2",
        edges=(_edge(2, 30, 40), _edge(3, 40, 50)),
        path_score=1.0,
        path_rank=2,
    )
    unselected = _PathCandidate(
        component_id=2,
        fragment_row=(100, 200, 300, -1, -1),
        fragment_span="0-2",
        edges=(_edge(2, 300, 400),),
        path_score=0.5,
        path_rank=3,
    )

    row = exposure._exposure_row("jm000", (selected, unselected), (selected,))

    assert row == {
        "subject": "jm000",
        "n_suffix_fragments": 2,
        "n_candidate_paths": 2,
        "n_selected_stitches": 1,
        "selected_path_lengths": "2",
        "selected_stitches_per_subject": 1,
    }


def test_exposure_aggregate_reports_selected_stitches_per_subject() -> None:
    rows = (
        {
            "subject": "jm000",
            "n_suffix_fragments": 2,
            "n_candidate_paths": 10,
            "n_selected_stitches": 1,
            "selected_path_lengths": "2",
            "selected_stitches_per_subject": 1,
        },
        {
            "subject": "jm001",
            "n_suffix_fragments": 3,
            "n_candidate_paths": 20,
            "n_selected_stitches": 0,
            "selected_path_lengths": "",
            "selected_stitches_per_subject": 0,
        },
    )

    row = exposure._aggregate_row(rows)

    assert row == {
        "subject": "ALL",
        "n_suffix_fragments": 5,
        "n_candidate_paths": 30,
        "n_selected_stitches": 1,
        "selected_path_lengths": "2",
        "selected_stitches_per_subject": "jm000:1;jm001:0",
    }


def _selected_path_ids(
    paths: tuple[_PathCandidate, ...],
) -> tuple[tuple[int, tuple[tuple[int, int, int, int], ...]], ...]:
    return tuple(
        (
            int(path.component_id),
            tuple(tuple(int(value) for value in edge.edge) for edge in path.edges),
        )
        for path in paths
    )
