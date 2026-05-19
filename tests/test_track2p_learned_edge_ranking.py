from __future__ import annotations

import csv

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_learned_edge_ranking import (
    run_track2p_learned_edge_ranking,
)
from tests.test_track2p_benchmark import _write_ground_truth_csv, _write_subject


def test_track2p_learned_edge_ranking_writes_monotone_loso_scores(
    tmp_path, monkeypatch, write_raw_npy_session
):
    session_names = ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a")
    for subject_name in ("jm001", "jm002"):
        subject_dir = tmp_path / subject_name
        _write_subject(subject_dir, write_raw_npy_session, write_reference=False)
        _write_ground_truth_csv(
            subject_dir,
            session_names,
            ((0, 0, 0), (1, 1, 1)),
        )

    from bayescatrack.association import calibrated_costs

    monkeypatch.setattr(
        calibrated_costs,
        "register_plane_pair",
        lambda _reference, moving, **_kwargs: moving,
    )

    output_path = tmp_path / "learned_edge_ranking.csv"
    summary_path = tmp_path / "learned_edge_ranking_summary.csv"
    edge_rows, summary_rows = run_track2p_learned_edge_ranking(
        Track2pBenchmarkConfig(
            data=tmp_path,
            method="global-assignment",
            split="leave-one-subject-out",
            cost="registered-iou",
            max_gap=2,
        ),
        output_path,
        summary_output_path=summary_path,
        score_model="monotone",
        feature_names=("centroid_distance", "one_minus_iou"),
    )

    assert edge_rows == 36
    assert summary_rows == 18

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["subject"] for row in rows} == {"jm001", "jm002"}
    assert {row["score_name"] for row in rows} == {
        "monotone_cost",
        "monotone_probability",
        "monotone_raw_score",
    }
    probability_directions = {
        row["score_direction"]
        for row in rows
        if row["score_name"] == "monotone_probability"
    }
    assert probability_directions == {"similarity"}
    assert all(row["score_model"] == "monotone" for row in rows)
    assert all(
        row["model_feature_names"] == "centroid_distance,one_minus_iou"
        for row in rows
    )

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == summary_rows
