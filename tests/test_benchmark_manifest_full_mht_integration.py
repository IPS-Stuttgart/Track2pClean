from __future__ import annotations

import csv
import json
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
    assert "cell_probability_threshold" not in kwargs
    assert "max_gap" not in kwargs


def test_full_mht_manifest_dispatches_prior_veto_options(tmp_path, monkeypatch) -> None:
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
                    "name": "FullMHTPriorVetoScaled",
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
                }
            ],
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert [run.name for run in result.runs] == ["FullMHTPriorVetoScaled"]
    rows = _read_csv_rows(tmp_path / "results" / "full_mht.csv")
    assert rows[0]["subject"] == "jm_fake"
    assert captured["config"].method == "global-assignment"
    assert captured["config"].max_gap == 3
    assert captured["threshold_method"] == "min"
    assert captured["iou_distance_threshold"] == 12.0
    assert captured["transform_type"] == "affine"
    assert captured["cell_probability_threshold"] == 0.5
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
