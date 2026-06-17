from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.advanced_uncertainty import (
    EdgeUncertaintyConfig,
    edge_uncertainty_config_from_mapping,
    uncertainty_aware_cost_matrix,
)
from bayescatrack.association.teacher_priors import (
    TeacherEdgePriorConfig,
    apply_teacher_edge_priors,
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
            "--track2p-teacher-prior-json",
            '{"relief": 0.75, "teacher_cost_cap": 0.5}',
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
    assert config.track2p_teacher_prior_config == {
        "relief": 0.75,
        "teacher_cost_cap": 0.5,
    }


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", True),
        ("temperature", False),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", 0.0),
        ("uncertainty_penalty_weight", True),
        ("uncertainty_penalty_weight", False),
        ("uncertainty_penalty_weight", float("nan")),
        ("uncertainty_penalty_weight", float("inf")),
        ("uncertainty_penalty_weight", -0.1),
        ("registration_rmse_weight", True),
        ("registration_rmse_weight", False),
        ("registration_rmse_weight", float("nan")),
        ("registration_rmse_weight", float("inf")),
        ("registration_rmse_weight", -0.1),
        ("invalid_warp_fraction_weight", True),
        ("invalid_warp_fraction_weight", False),
        ("invalid_warp_fraction_weight", float("nan")),
        ("invalid_warp_fraction_weight", float("inf")),
        ("invalid_warp_fraction_weight", -0.1),
        ("empty_registered_roi_weight", True),
        ("empty_registered_roi_weight", False),
        ("empty_registered_roi_weight", float("nan")),
        ("empty_registered_roi_weight", float("inf")),
        ("empty_registered_roi_weight", -0.1),
        ("gated_edge_weight", True),
        ("gated_edge_weight", False),
        ("gated_edge_weight", float("nan")),
        ("gated_edge_weight", float("inf")),
        ("gated_edge_weight", -0.1),
        ("covariance_logdet_weight", True),
        ("covariance_logdet_weight", False),
        ("covariance_logdet_weight", float("nan")),
        ("covariance_logdet_weight", float("inf")),
        ("covariance_logdet_weight", -0.1),
        ("local_margin_weight", True),
        ("local_margin_weight", False),
        ("local_margin_weight", float("nan")),
        ("local_margin_weight", float("inf")),
        ("local_margin_weight", -0.1),
        ("activity_missing_weight", True),
        ("activity_missing_weight", False),
        ("activity_missing_weight", float("nan")),
        ("activity_missing_weight", float("inf")),
        ("activity_missing_weight", -0.1),
        ("min_reliability", True),
        ("min_reliability", False),
        ("min_reliability", float("nan")),
        ("min_reliability", float("inf")),
        ("min_reliability", 0.0),
        ("min_reliability", 1.1),
        ("max_penalty", True),
        ("max_penalty", False),
        ("max_penalty", float("nan")),
        ("max_penalty", float("inf")),
        ("max_penalty", 0.0),
    ],
)
def test_edge_uncertainty_config_rejects_invalid_controls(
    field: str, value: float | bool
) -> None:
    with pytest.raises(ValueError, match=field):
        EdgeUncertaintyConfig(**{field: value})


def test_track2p_teacher_prior_reliefs_suite2p_edges() -> None:
    sessions = (
        _fake_session([10, 11]),
        _fake_session([20, 21]),
    )
    adjusted = apply_teacher_edge_priors(
        {(0, 1): np.full((2, 2), 5.0, dtype=float)},
        sessions,
        teacher_track_matrix=np.array([[10, 21], [11, 20]], dtype=int),
        config=TeacherEdgePriorConfig(relief=1.0, teacher_cost_cap=2.0, min_cost=0.0),
    )

    assert adjusted[(0, 1)][0, 1] == pytest.approx(1.0)
    assert adjusted[(0, 1)][1, 0] == pytest.approx(1.0)
    assert adjusted[(0, 1)][0, 0] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relief", True),
        ("relief", False),
        ("relief", float("nan")),
        ("relief", float("inf")),
        ("relief", -0.1),
        ("teacher_cost_cap", True),
        ("teacher_cost_cap", False),
        ("teacher_cost_cap", float("nan")),
        ("teacher_cost_cap", float("inf")),
        ("non_teacher_penalty", True),
        ("non_teacher_penalty", False),
        ("non_teacher_penalty", float("nan")),
        ("non_teacher_penalty", float("inf")),
        ("non_teacher_penalty", -0.1),
        ("min_cost", True),
        ("min_cost", False),
        ("min_cost", float("nan")),
        ("min_cost", float("inf")),
        ("large_cost", True),
        ("large_cost", False),
        ("large_cost", float("nan")),
        ("large_cost", float("inf")),
        ("large_cost", 0.0),
    ],
)
def test_teacher_edge_prior_config_rejects_invalid_float_controls(
    field: str, value: float | bool
) -> None:
    with pytest.raises(ValueError, match=field):
        TeacherEdgePriorConfig(**{field: value})


@pytest.mark.parametrize(
    "max_gap",
    [True, False, 0, -1, 1.5, "2", float("nan"), float("inf")],
)
def test_teacher_edge_prior_config_rejects_invalid_max_gap(max_gap: object) -> None:
    with pytest.raises(ValueError, match="max_gap"):
        TeacherEdgePriorConfig(max_gap=max_gap)  # type: ignore[arg-type]


def _fake_session(roi_indices: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        plane_data=SimpleNamespace(
            n_rois=len(roi_indices), roi_indices=np.asarray(roi_indices, dtype=int)
        )
    )


def test_improvement_manifest_includes_diagnostics_priors_and_uncertainty() -> None:
    manifest = track2p_result_improvement_manifest(
        data_root="data", reference_root="gt", output_root="results"
    )

    assert manifest["defaults"]["seed_sessions"] == "all"
    assert manifest["defaults"]["fov_affine_mask_warp_mode"] == "bilinear"
    run_names = [run["name"] for run in manifest["runs"]]
    assert len(run_names) == len(set(run_names))
    run_name_set = set(run_names)
    assert "track2p-policy-dp" in run_name_set
    assert "track2p-policy-pruned" in run_name_set
    assert "oracle-gt-links" in run_name_set
    assert "roi-aware-shifted-auto-registration" in run_name_set
    assert "roi-aware-shifted-dynamic-priors" in run_name_set
    assert "roi-aware-shifted-uncertainty-pruned" in run_name_set
    assert "roi-aware-shifted-learned-solver-priors" in run_name_set
    assert "roi-aware-shifted-track2p-teacher-prior" in run_name_set
    assert any("candidate_pruning_config" in run for run in manifest["runs"])
    assert any("dynamic_edge_prior_config" in run for run in manifest["runs"])
    assert any("edge_uncertainty_config" in run for run in manifest["runs"])
    assert any("track2p_teacher_prior_config" in run for run in manifest["runs"])
    assert any(run.get("objective") == "complete_track_f1" for run in manifest["runs"])
