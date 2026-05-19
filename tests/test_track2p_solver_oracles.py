from __future__ import annotations

import pytest
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_solver_oracles import (
    parse_rank_ks,
    run_track2p_solver_oracles,
)
from tests.test_track2p_benchmark import (
    _install_fake_multisession_assignment,
    _write_ground_truth_csv,
    _write_subject,
)


def test_parse_rank_ks_deduplicates_and_validates_values():
    assert parse_rank_ks("1,3,3,10") == (1, 3, 10)
    with pytest.raises(ValueError, match="positive integers"):
        parse_rank_ks("1,0")
    with pytest.raises(ValueError, match="non-integer"):
        parse_rank_ks("1,bad")


def test_solver_oracle_harness_scores_gt_edge_costs(
    tmp_path, monkeypatch, write_raw_npy_session
):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session, write_reference=False)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a"),
        ((0, 0, 0), (1, -1, 1)),
    )
    _install_fake_multisession_assignment(monkeypatch)

    results = run_track2p_solver_oracles(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="global-assignment",
            reference_kind="manual-gt",
            max_gap=2,
        ),
        oracles=("edge-costs",),
    )

    assert len(results) == 1
    result = results[0].to_dict()
    assert result["variant"] == "Solver oracle: GT edge costs"
    assert result["oracle"] == "edge-costs"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)
