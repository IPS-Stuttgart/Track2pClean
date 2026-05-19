from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bayescatrack.experiments import track2p_benchmark
from bayescatrack.reference import Track2pReference

# pylint: disable=protected-access


def _session(session_name: str, suite2p_indices: list[int]):
    return SimpleNamespace(
        session_name=session_name,
        plane_data=SimpleNamespace(
            n_rois=len(suite2p_indices),
            roi_indices=np.asarray(suite2p_indices, dtype=int),
        ),
    )


def test_cli_accepts_oracle_gt_solver_methods():
    parser = track2p_benchmark.build_arg_parser()

    args = parser.parse_args(
        ["--data", "dataset", "--method", "oracle-gt-solver"]
    )
    config = track2p_benchmark._config_from_args(args)
    assert config.method == "oracle-gt-solver"

    args = parser.parse_args(
        ["--data", "dataset", "--method", "oracle-gt-consecutive-solver"]
    )
    config = track2p_benchmark._config_from_args(args)
    assert config.method == "oracle-gt-consecutive-solver"


def test_oracle_pairwise_costs_convert_suite2p_indices_to_loaded_positions():
    reference = Track2pReference(
        session_names=("s0", "s1", "s2"),
        suite2p_indices=np.asarray(
            [
                [10, 20, 30],
                [11, None, 31],
            ],
            dtype=object,
        ),
    )
    sessions = [
        _session("s0", [10, 11]),
        _session("s1", [99, 20]),
        _session("s2", [31, 30]),
    ]

    costs = track2p_benchmark._oracle_pairwise_costs_from_reference(
        reference,
        sessions,
        max_gap=2,
        curated_only=False,
        match_cost=0.0,
        nonmatch_cost=99.0,
    )

    assert costs[(0, 1)].shape == (2, 2)
    assert costs[(0, 1)][0, 1] == 0.0  # Suite2p ROI 10 -> 20.
    assert costs[(0, 1)][1, 0] == 99.0  # ROI 11 has no GT partner in s1.

    assert costs[(0, 2)].shape == (2, 2)
    assert costs[(0, 2)][0, 1] == 0.0  # Suite2p ROI 10 -> 30.
    assert costs[(0, 2)][1, 0] == 0.0  # Suite2p ROI 11 -> 31.

    assert costs[(1, 2)].shape == (2, 2)
    assert costs[(1, 2)][1, 1] == 0.0  # Suite2p ROI 20 -> 30.
    assert costs[(1, 2)][0, 0] == 99.0


def test_oracle_pairwise_costs_reject_missing_filtered_reference_roi():
    reference = Track2pReference(
        session_names=("s0", "s1"),
        suite2p_indices=np.asarray([[10, 20]], dtype=object),
    )
    sessions = [_session("s0", [10]), _session("s1", [21])]

    with np.testing.assert_raises_regex(ValueError, "absent from the loaded ROI set"):
        track2p_benchmark._oracle_pairwise_costs_from_reference(
            reference,
            sessions,
            max_gap=1,
            curated_only=False,
            match_cost=0.0,
            nonmatch_cost=99.0,
        )
