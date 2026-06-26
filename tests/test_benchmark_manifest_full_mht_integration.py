from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayescatrack.experiments import benchmark_manifest as bm
from bayescatrack.experiments.benchmark_manifest import (
    load_benchmark_manifest,
    run_benchmark_manifest,
)


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _read_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_full_mht_canonical_manifest_keeps_greedy_beam_ablation() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_prior_veto_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "FullMHTGreedyPrior2",
        "FullMHTPriorVetoScaled",
        "FullMHTPriorSurvival",
    ]
    greedy = runs["FullMHTGreedyPrior2"]
    beam = runs["FullMHTPrior2"]
    assert greedy["runner"] == "track2p-full-mht"
    assert greedy["beam_width"] == 1
    assert greedy["scan_hypotheses"] == beam["scan_hypotheses"]
    assert greedy["edge_top_k"] == beam["edge_top_k"]
    assert greedy["track2p_prior_weight"] == beam["track2p_prior_weight"]
    assert greedy["track2p_non_prior_penalty"] == beam["track2p_non_prior_penalty"]
    assert greedy["track2p_prior_miss_penalty"] == beam["track2p_prior_miss_penalty"]

    for comparison in manifest["comparisons"]:
        assert comparison["inputs"]["FullMHTGreedyPrior2"] == "FullMHTGreedyPrior2"


def test_full_mht_terminal_completion_probe_manifest_is_frozen() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_terminal_completion_probe_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "FullMHTTerminalCompletion025",
        "FullMHTTerminalCompletion050",
        "FullMHTTerminalCompletion100",
    ]
    assert runs["FullMHTTerminalCompletion025"]["terminal_incomplete_history_weight"] == 0.25
    assert runs["FullMHTTerminalCompletion050"]["terminal_incomplete_history_weight"] == 0.50
    assert runs["FullMHTTerminalCompletion100"]["terminal_incomplete_history_weight"] == 1.00
    for name in (
        "FullMHTTerminalCompletion025",
        "FullMHTTerminalCompletion050",
        "FullMHTTerminalCompletion100",
    ):
        assert runs[name]["runner"] == "track2p-full-mht"
        assert runs[name]["beam_width"] == runs["FullMHTPrior2"]["beam_width"]
        assert runs[name]["track2p_prior_weight"] == runs["FullMHTPrior2"]["track2p_prior_weight"]


def test_full_mht_history_dynamics_probe_manifest_is_frozen() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_history_dynamics_probe_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "FullMHTHistoryDynamics025",
        "FullMHTHistoryDynamics050",
        "FullMHTHistoryDynamics100",
    ]
    assert runs["FullMHTHistoryDynamics025"]["terminal_motion_history_weight"] == 0.25
    assert runs["FullMHTHistoryDynamics050"]["terminal_motion_history_weight"] == 0.50
    assert runs["FullMHTHistoryDynamics100"]["terminal_motion_history_weight"] == 1.00
    for name in (
        "FullMHTHistoryDynamics025",
        "FullMHTHistoryDynamics050",
        "FullMHTHistoryDynamics100",
    ):
        assert runs[name]["runner"] == "track2p-full-mht"
        assert runs[name]["beam_width"] == runs["FullMHTPrior2"]["beam_width"]
        assert runs[name]["track2p_prior_weight"] == runs["FullMHTPrior2"]["track2p_prior_weight"]


def test_full_mht_scan_history_dynamics_probe_manifest_is_frozen() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_scan_history_dynamics_probe_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "FullMHTScanHistoryDynamics025",
        "FullMHTScanHistoryDynamics050",
        "FullMHTScanHistoryDynamics100",
    ]
    assert runs["FullMHTScanHistoryDynamics025"]["scan_motion_history_weight"] == 0.25
    assert runs["FullMHTScanHistoryDynamics050"]["scan_motion_history_weight"] == 0.50
    assert runs["FullMHTScanHistoryDynamics100"]["scan_motion_history_weight"] == 1.00
    for name in (
        "FullMHTScanHistoryDynamics025",
        "FullMHTScanHistoryDynamics050",
        "FullMHTScanHistoryDynamics100",
    ):
        assert runs[name]["runner"] == "track2p-full-mht"
        assert runs[name]["beam_width"] == runs["FullMHTPrior2"]["beam_width"]
        assert runs[name]["track2p_prior_weight"] == runs["FullMHTPrior2"]["track2p_prior_weight"]


def test_full_mht_growth_history_prediction_probe_manifest_is_frozen() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_growth_history_prediction_probe_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "FullMHTGrowthHistoryPrediction025",
        "FullMHTGrowthHistoryPrediction050",
        "FullMHTGrowthHistoryPrediction100",
    ]
    assert runs["FullMHTGrowthHistoryPrediction025"]["growth_history_prediction_weight"] == 0.25
    assert runs["FullMHTGrowthHistoryPrediction050"]["growth_history_prediction_weight"] == 0.50
    assert runs["FullMHTGrowthHistoryPrediction100"]["growth_history_prediction_weight"] == 1.00
    for name in (
        "FullMHTGrowthHistoryPrediction025",
        "FullMHTGrowthHistoryPrediction050",
        "FullMHTGrowthHistoryPrediction100",
    ):
        assert runs[name]["runner"] == "track2p-full-mht"
        assert runs[name]["beam_width"] == runs["FullMHTPrior2"]["beam_width"]
        assert runs[name]["track2p_prior_weight"] == runs["FullMHTPrior2"]["track2p_prior_weight"]
        assert runs[name]["growth_history_prediction_scale"] == 1.0
        assert runs[name]["growth_history_prediction_clip"] == 8.0
        assert runs[name]["growth_history_prediction_min_edges"] == 1


def test_full_mht_manifest_runner_aliases_are_supported() -> None:
    assert bm._runner_name("track2p-policy-full-mht") == "track2p-policy-full-mht"
    assert bm._runner_name("track2p-full-mht") == "track2p-policy-full-mht"
    assert bm._runner_name("track2p-pyrecest-full-mht") == "track2p-policy-full-mht"


def test_full_mht_runner_kwargs_keep_mht_gap_separate_from_track2p_config() -> None:
    kwargs = bm._runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "max_gap": 3,
            "full_mht_max_gap": 1,
            "beam_width": 4,
            "scan_hypotheses": 4,
            "track2p_prior_weight": 12.0,
            "track2p_prior_veto_penalty": 20.0,
            "track2p_prior_veto_min_growth_residual_mahalanobis": 2.5,
            "track2p_prior_veto_max_registered_iou": 0.40,
            "track2p_prior_survival_weight": 1.75,
            "track2p_prior_survival_min_anchor_registered_iou": 0.82,
            "track2p_prior_survival_min_examples_per_class": 3,
            "track2p_prior_survival_score_clip": 5.0,
            "terminal_incomplete_history_weight": 0.75,
            "terminal_motion_history_weight": 0.50,
            "scan_motion_history_weight": 0.40,
            "growth_history_prediction_weight": 0.60,
            "growth_history_prediction_scale": 2.0,
            "growth_history_prediction_clip": 4.0,
            "growth_history_prediction_min_edges": 2,
        },
        "track2p-policy-full-mht",
    )

    assert kwargs["threshold_method"] == "min"
    assert kwargs["iou_distance_threshold"] == 12.0
    assert kwargs["full_mht_max_gap"] == 1
    assert kwargs["beam_width"] == 4
    assert kwargs["scan_hypotheses"] == 4
    assert kwargs["track2p_prior_weight"] == 12.0
    assert kwargs["track2p_prior_veto_penalty"] == 20.0
    assert kwargs["track2p_prior_veto_min_growth_residual_mahalanobis"] == 2.5
    assert kwargs["track2p_prior_veto_max_registered_iou"] == 0.40
    assert kwargs["track2p_prior_survival_weight"] == 1.75
    assert kwargs["track2p_prior_survival_min_anchor_registered_iou"] == 0.82
    assert kwargs["track2p_prior_survival_min_examples_per_class"] == 3
    assert kwargs["track2p_prior_survival_score_clip"] == 5.0
    assert kwargs["terminal_incomplete_history_weight"] == 0.75
    assert kwargs["terminal_motion_history_weight"] == 0.50
    assert kwargs["scan_motion_history_weight"] == 0.40
    assert kwargs["growth_history_prediction_weight"] == 0.60
    assert kwargs["growth_history_prediction_scale"] == 2.0
    assert kwargs["growth_history_prediction_clip"] == 4.0
    assert kwargs["growth_history_prediction_min_edges"] == 2
    assert "cell_probability_threshold" not in kwargs
    assert "max_gap" not in kwargs


def test_full_mht_manifest_dispatches_prior_veto_survival_completion_and_history_options(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("pyrecest")
    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    captured = {}

    class FakeResult:
        def to_dict(self):
            return {
                "subject": "jm_fake",
                "method": "track2p-policy-full-mht",
                "pairwise_f1": 1.0,
                "complete_track_f1": 1.0,
            }

    def fake_run(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return SimpleNamespace(results=(FakeResult(),))

    monkeypatch.setattr(full_mht, "run_track2p_policy_full_mht", fake_run)
    manifest_path = tmp_path / "full_mht_manifest.json"
    _write_manifest(
        manifest_path,
        {
            "defaults": {
                "data": "data-root",
                "reference": "reference-root",
                "reference_kind": "manual-gt",
                "input_format": "suite2p",
                "threshold_method": "min",
                "transform_type": "affine",
                "iou_distance_threshold": 12.0,
                "cell_probability_threshold": 0.5,
                "max_gap": 3,
            },
            "runs": [
                {
                    "name": "FullMHTPriorSurvivalCompletionHistory",
                    "runner": "track2p-full-mht",
                    "output": "results/full_mht.csv",
                    "beam_width": 4,
                    "scan_hypotheses": 4,
                    "edge_top_k": 3,
                    "identity_diverse_beam": True,
                    "miss_cost": 2.0,
                    "full_mht_max_gap": 1,
                    "gap_reactivation_cost": 1.5,
                    "min_output_observations": 2,
                    "min_edge_score": 0.25,
                    "seed_source": "reference",
                    "association_score_mode": "heuristic",
                    "track2p_prior_weight": 12.0,
                    "track2p_non_prior_penalty": 2.0,
                    "track2p_prior_switch_penalty": 8.0,
                    "track2p_no_prior_successor_penalty": 8.0,
                    "track2p_prior_miss_penalty": 4.0,
                    "track2p_prior_veto_penalty": 20.0,
                    "track2p_prior_veto_min_growth_residual_mahalanobis": 2.5,
                    "track2p_prior_veto_min_growth_residual": 2.5,
                    "track2p_prior_veto_min_registered_iou": 0.35,
                    "track2p_prior_veto_max_registered_iou": 0.40,
                    "track2p_prior_veto_min_shifted_iou": 0.60,
                    "track2p_prior_veto_max_shifted_iou": 0.80,
                    "track2p_prior_veto_min_cell_probability": 0.50,
                    "track2p_prior_veto_max_min_cell_probability": 0.65,
                    "track2p_prior_veto_max_row_rank": 1,
                    "track2p_prior_veto_max_column_rank": 1,
                    "track2p_prior_veto_require_terminal_edge": True,
                    "track2p_prior_veto_require_last_session_edge": True,
                    "track2p_prior_veto_require_complete_component": True,
                    "track2p_prior_survival_weight": 1.75,
                    "track2p_prior_survival_min_anchor_registered_iou": 0.82,
                    "track2p_prior_survival_min_anchor_shifted_iou": 0.72,
                    "track2p_prior_survival_max_anchor_growth_mahalanobis": 1.25,
                    "track2p_prior_survival_max_anchor_growth_residual": 1.10,
                    "track2p_prior_survival_min_anchor_cell_probability": 0.84,
                    "track2p_prior_survival_max_anchor_rank": 1,
                    "track2p_prior_survival_max_background_registered_iou": 0.45,
                    "track2p_prior_survival_max_background_shifted_iou": 0.55,
                    "track2p_prior_survival_min_background_growth_mahalanobis": 2.2,
                    "track2p_prior_survival_min_background_growth_residual": 2.0,
                    "track2p_prior_survival_max_background_cell_probability": 0.70,
                    "track2p_prior_survival_min_examples_per_class": 3,
                    "track2p_prior_survival_min_feature_scale": 0.07,
                    "track2p_prior_survival_per_feature_clip": 3.5,
                    "track2p_prior_survival_score_clip": 5.0,
                    "terminal_incomplete_history_weight": 0.75,
                    "terminal_motion_history_weight": 0.50,
                    "scan_motion_history_weight": 0.40,
                    "growth_history_prediction_weight": 0.60,
                    "growth_history_prediction_scale": 2.0,
                    "growth_history_prediction_clip": 4.0,
                    "growth_history_prediction_min_edges": 2,
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert [run.name for run in result.runs] == ["FullMHTPriorSurvivalCompletionHistory"]
    rows = _read_csv_rows(tmp_path / "results" / "full_mht.csv")
    assert rows[0]["subject"] == "jm_fake"
    assert captured["config"].method == "global-assignment"
    assert captured["config"].max_gap == 3
    assert captured["threshold_method"] == "min"
    assert captured["iou_distance_threshold"] == 12.0
    assert captured["transform_type"] == "affine"
    assert captured["cell_probability_threshold"] == 0.5
    assert getattr(full_mht, "_bayescatrack_prior_survival_scoring", False)
    assert getattr(full_mht, "_bayescatrack_terminal_completion_objective", False)
    assert getattr(full_mht, "_bayescatrack_history_dynamics_objective", False)
    assert getattr(full_mht, "_bayescatrack_scan_history_dynamics_pruning", False)
    assert getattr(full_mht, "_bayescatrack_growth_history_prediction_scoring", False)
    mht_config = captured["mht_config"]
    assert mht_config.beam_width == 4
    assert mht_config.scan_hypotheses == 4
    assert mht_config.edge_top_k == 3
    assert mht_config.identity_diverse_beam is True
    assert mht_config.max_gap == 1
    assert mht_config.min_output_observations == 2
    assert mht_config.track2p_prior_weight == 12.0
    assert mht_config.track2p_non_prior_penalty == 2.0
    assert mht_config.track2p_prior_miss_penalty == 4.0
    assert mht_config.track2p_prior_veto_penalty == 20.0
    assert mht_config.track2p_prior_veto_min_growth_residual_mahalanobis == 2.5
    assert mht_config.track2p_prior_veto_min_registered_iou == 0.35
    assert mht_config.track2p_prior_veto_max_registered_iou == 0.40
    assert mht_config.track2p_prior_veto_max_min_cell_probability == 0.65
    assert getattr(mht_config, "track2p_prior_survival_weight") == 1.75
    assert getattr(
        mht_config, "track2p_prior_survival_min_anchor_registered_iou"
    ) == 0.82
    assert getattr(mht_config, "track2p_prior_survival_max_anchor_rank") == 1
    assert getattr(mht_config, "track2p_prior_survival_min_examples_per_class") == 3
    assert getattr(mht_config, "track2p_prior_survival_score_clip") == 5.0
    assert getattr(mht_config, "terminal_incomplete_history_weight") == 0.75
    assert getattr(mht_config, "terminal_motion_history_weight") == 0.50
    assert getattr(mht_config, "scan_motion_history_weight") == 0.40
    assert getattr(mht_config, "growth_history_prediction_weight") == 0.60
    assert getattr(mht_config, "growth_history_prediction_scale") == 2.0
    assert getattr(mht_config, "growth_history_prediction_clip") == 4.0
    assert getattr(mht_config, "growth_history_prediction_min_edges") == 2
