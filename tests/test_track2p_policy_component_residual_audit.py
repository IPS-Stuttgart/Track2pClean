from __future__ import annotations

import numpy as np
from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_component_residual_audit as audit


def test_component_residual_audit_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-component-residual-audit"]

    assert canonical == "track2p-policy-component-residual-audit"
    assert cli._BENCHMARK_ALIASES["track2p-residual-audit"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_component_residual_audit"
    )


def test_residual_error_rows_are_duplicate_aware() -> None:
    predicted = np.asarray(
        [
            [1, 2, 3],
            [1, 2, 4],
        ],
        dtype=int,
    )
    reference = np.asarray([[1, 2, 3]], dtype=int)

    rows = audit.residual_error_rows(
        predicted,
        reference,
        subject="subject-a",
        policy_tracks=predicted,
        before_cleanup_tracks=predicted,
    )

    counts = _counts_by_error_type(rows)
    assert counts == {
        "pairwise_fp": 2,
        "pairwise_fn": 0,
        "complete_fp": 1,
        "complete_fn": 0,
    }
    fp_edge = next(row for row in rows if row["track_id_or_edge"] == "1:2->2:4")
    assert fp_edge["track_id_or_edge"] == "1:2->2:4"
    assert fp_edge["reason_bucket"] == "wrong target selected"


def test_residual_error_rows_classify_missing_adjacent_edge() -> None:
    predicted = np.asarray([[10, 11, -1]], dtype=int)
    reference = np.asarray([[10, 11, 12]], dtype=int)
    gap_tracks = np.asarray([[10, 11, 12]], dtype=int)

    rows = audit.residual_error_rows(
        predicted,
        reference,
        subject="subject-a",
        gap_tracks=gap_tracks,
    )

    counts = _counts_by_error_type(rows)
    assert counts["pairwise_fn"] == 1
    assert counts["complete_fn"] == 1
    fn_edge = next(row for row in rows if row["error_type"] == "pairwise_fn")
    assert fn_edge["track_id_or_edge"] == "1:11->2:12"
    assert fn_edge["is_gap_rescue_supported"] == 1
    assert (
        fn_edge["reason_bucket"]
        == "gap evidence not converted into adjacent scored edge"
    )


def test_component_residual_parser_defaults_to_component_cleanup_row() -> None:
    args = audit.build_arg_parser().parse_args(
        ["--data", "track2p-root", "--output", "x.csv"]
    )

    assert args.threshold_method == "min"
    assert args.iou_distance_threshold == 12
    assert args.cell_probability_threshold == 0.5
    assert args.split_risk_threshold == 1.5
    assert args.min_side_observations == 2
    assert args.feature_mode == "policy-diagnostics"


def _counts_by_error_type(
    rows: list[dict[str, float | int | str]],
) -> dict[str, int]:
    counts = {
        "pairwise_fp": 0,
        "pairwise_fn": 0,
        "complete_fp": 0,
        "complete_fn": 0,
    }
    for row in rows:
        counts[str(row["error_type"])] += 1
    return counts
