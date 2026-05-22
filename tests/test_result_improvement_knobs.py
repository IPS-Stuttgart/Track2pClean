from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bayescatrack.association.advanced_uncertainty import (
    EdgeUncertaintyConfig,
    edge_uncertainty_config_from_mapping,
    uncertainty_aware_cost_matrix,
)
from bayescatrack.experiments.advanced_improvement_workbench import (
    track2p_result_improvement_manifest,
)
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _config_from_args,
    _resolved_seed_sessions,
    build_arg_parser,
)
from bayescatrack.registration_selection import DEFAULT_AUTO_REGISTRATION_CANDIDATES


def test_auto_registration_default_includes_fov_affine() -> None:
    assert "fov-affine" in DEFAULT_AUTO_REGISTRATION_CANDIDATES


def test_seed_sessions_all_resolves_all_sessions() -> None:
    config = Track2pBenchmarkConfig(
        data=Path("."),
        method="track2p-baseline",
        seed_sessions="all",
    )

    assert _resolved_seed_sessions(config, n_sessions=4) == (0, 1, 2, 3)


def test_track2p_parser_exposes_result_improvement_knobs() -> None:
    args = build_arg_parser().parse_args(
        [
            "--data",
            ".",
            "--method",
            "global-assignment",
            "--reference-kind",
            "manual-gt",
            "--transform-type",
            "auto",
            "--auto-registration-candidates",
            "none,fov-affine,local-affine-grid",
            "--fov-affine-mask-warp-mode",
            "bilinear",
            "--seed-sessions",
            "all",
            "--edge-uncertainty-json",
            '{"uncertainty_penalty_weight": 0.5}',
        ]
    )

    config = _config_from_args(args)

    assert config.seed_sessions == "all"
    assert config.auto_registration_candidates == (
        "none",
        "fov-affine",
        "local-affine-grid",
    )
    assert config.fov_affine_mask_warp_mode == "bilinear"
    assert config.edge_uncertainty_config == {"uncertainty_penalty_weight": 0.5}


def test_uncertainty_mapping_penalizes_unreliable_edges() -> None:
    config = edge_uncertainty_config_from_mapping(
        {"uncertainty_penalty_weight": 1.0, "gated_edge_weight": 3.0}
    )

    assert isinstance(config, EdgeUncertaintyConfig)
    result = uncertainty_aware_cost_matrix(
        np.zeros((1, 2), dtype=float),
        {"gated": np.array([[False, True]])},
        config=config,
    )

    assert result.adjusted_cost_matrix[0, 0] == pytest.approx(0.0)
    assert result.adjusted_cost_matrix[0, 1] > result.adjusted_cost_matrix[0, 0]
    assert result.reliability_matrix[0, 1] < result.reliability_matrix[0, 0]


def test_uncertainty_penalizes_empty_registered_roi_columns() -> None:
    result = uncertainty_aware_cost_matrix(
        np.zeros((2, 2), dtype=float),
        {},
        empty_registered_rois=np.array([False, True]),
        config=EdgeUncertaintyConfig(
            uncertainty_penalty_weight=1.0,
            empty_registered_roi_weight=3.0,
        ),
    )

    assert np.all(result.adjusted_cost_matrix[:, 1] > result.adjusted_cost_matrix[:, 0])


def test_improvement_manifest_includes_diagnostics_priors_and_uncertainty() -> None:
    manifest = track2p_result_improvement_manifest(
        data_root="data", reference_root="gt", output_root="results"
    )

    assert manifest["defaults"]["seed_sessions"] == "all"
    assert manifest["defaults"]["fov_affine_mask_warp_mode"] == "bilinear"
    run_names = {run["name"] for run in manifest["runs"]}
    assert "oracle-gt-links" in run_names
    assert "roi-aware-shifted-auto-registration" in run_names
    assert "roi-aware-shifted-dynamic-priors" in run_names
    assert "roi-aware-shifted-uncertainty-pruned" in run_names
    assert "roi-aware-shifted-learned-solver-priors" in run_names
    assert any("candidate_pruning_config" in run for run in manifest["runs"])
    assert any("dynamic_edge_prior_config" in run for run in manifest["runs"])
    assert any("edge_uncertainty_config" in run for run in manifest["runs"])
    assert any(run.get("objective") == "complete_track_f1" for run in manifest["runs"])
