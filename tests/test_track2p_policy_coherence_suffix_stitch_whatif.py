import numpy as np

from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_coherence_suffix_stitch as method
from bayescatrack.experiments import track2p_policy_coherence_suffix_stitch_whatif as audit
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
    assert cli._BENCHMARK_ALIASES["track2p-component-coherence-suffix-stitch-whatif"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == ("bayescatrack.experiments.track2p_policy_coherence_suffix_stitch_whatif")


def test_coherence_suffix_stitch_method_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-coherence-suffix-stitch"]

    assert canonical == "track2p-policy-coherence-suffix-stitch"
    assert cli._BENCHMARK_ALIASES["track2p-component-coherence-suffix-stitch"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == "bayescatrack.experiments.track2p_policy_coherence_suffix_stitch"


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


def test_coherence_gate_accepts_two_edge_final_suffix() -> None:
    predicted = np.asarray([[1, 2, 3, -1, -1]], dtype=int)
    reference = np.asarray([[1, 2, 3, 4, 5]], dtype=int)
    path = _PathCandidate(
        component_id=0,
        fragment_row=(1, 2, 3, -1, -1),
        fragment_span="0-2",
        edges=(_edge(2, 3, 4), _edge(3, 4, 5)),
        path_score=0.5,
        path_rank=3,
        is_gt_suffix_path=1,
    )

    assert audit._passes_coherence_gate(path, predicted, reference, gate=audit.CoherenceSuffixStitchGate())


def test_coherence_gate_rejects_occupied_continuation() -> None:
    predicted = np.asarray([[1, 2, 3, 99, -1]], dtype=int)
    reference = np.asarray([[1, 2, 3, 4, 5]], dtype=int)
    path = _PathCandidate(
        component_id=0,
        fragment_row=(1, 2, 3, 99, -1),
        fragment_span="0-2",
        edges=(_edge(2, 3, 4), _edge(3, 4, 5)),
        path_score=0.5,
    )

    assert not audit._passes_coherence_gate(path, predicted, reference, gate=audit.CoherenceSuffixStitchGate())


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
    reference = np.asarray([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=int)
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
        reference,
        gate=audit.CoherenceSuffixStitchGate(max_stitches_per_subject=1),
    )

    assert selected == (stronger,)
