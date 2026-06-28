from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.advanced_uncertainty import (
    EdgeUncertaintyConfig,
    candidate_mask_from_posteriors,
    edge_uncertainty_config_from_mapping,
    posterior_probability_matrix,
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


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"seed_session": True}, "seed_session must contain integer session indices"),
        (
            {"seed_session": np.bool_(False)},
            "seed_session must contain integer session indices",
        ),
        ({"seed_session": 1.5}, "seed_session must contain integer session indices"),
        ({"seed_session": np.nan}, "seed_session must contain integer session indices"),
        (
            {"seed_sessions": (True,)},
            "seed_sessions must contain integer session indices",
        ),
        (
            {"seed_sessions": (np.bool_(False),)},
            "seed_sessions must contain integer session indices",
        ),
        (
            {"seed_sessions": (1.5,)},
            "seed_sessions must contain integer session indices",
        ),
    ],
)
def test_seed_session_resolver_rejects_silent_index_coercions(
    config_kwargs: dict[str, object],
    message: str,
) -> None:
    config = Track2pBenchmarkConfig(
        data=Path("."),
        method="track2p-baseline",
        **config_kwargs,
    )

    with pytest.raises(ValueError, match=message):
        _resolved_seed_sessions(config, n_sessions=4)


def test_seed_sessions_accept_numpy_integer_array_without_truth_coercion() -> None:
    config = Track2pBenchmarkConfig(
        data=Path("."),
        method="track2p-baseline",
        seed_sessions=np.asarray([1, 2], dtype=np.int64),
    )

    assert _resolved_seed_sessions(config, n_sessions=4) == (1, 2)


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


def test_uncertainty_config_rejects_nonfinite_runtime_values() -> None:
    with pytest.raises(ValueError, match="temperature must be finite"):
        EdgeUncertaintyConfig(temperature=np.nan)
    with pytest.raises(ValueError, match="registration_rmse_weight must be finite"):
        EdgeUncertaintyConfig(registration_rmse_weight=np.inf)
    with pytest.raises(ValueError, match="min_reliability must be finite"):
        EdgeUncertaintyConfig(min_reliability=np.nan)
    with pytest.raises(ValueError, match="max_penalty must be finite"):
        EdgeUncertaintyConfig(max_penalty=np.inf)
    with pytest.raises(ValueError, match="gated_edge_weight must be finite"):
        EdgeUncertaintyConfig(gated_edge_weight=True)


def test_posterior_probabilities_ignore_nonfinite_reliability_entries() -> None:
    probabilities = posterior_probability_matrix(
        np.asarray([[0.0, 1.0]], dtype=float),
        reliability_matrix=np.asarray([[1.0, np.nan]], dtype=float),
    )

    assert np.all(np.isfinite(probabilities))
    assert probabilities[0, 0] == pytest.approx(1.0)
    assert probabilities[0, 1] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_probability": np.nan}, "min_probability must be finite"),
        ({"min_probability": 1.5}, "min_probability must be a finite value"),
        ({"row_top_k": 1.5}, "row_top_k must be a positive integer"),
        ({"column_top_k": True}, "column_top_k must be finite"),
    ],
)
def test_candidate_mask_from_posteriors_rejects_invalid_pruning_knobs(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        candidate_mask_from_posteriors(np.asarray([[0.5, 0.25]]), **kwargs)


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


@pytest.mark.parametrize("boolean_value", [True, np.bool_(True)])
def test_track2p_teacher_prior_ignores_boolean_track_entries(
    boolean_value: object,
) -> None:
    sessions = (
        _fake_session([0, 1]),
        _fake_session([20]),
    )
    adjusted = apply_teacher_edge_priors(
        {(0, 1): np.full((2, 1), 5.0, dtype=float)},
        sessions,
        teacher_track_matrix=np.array([[boolean_value, 20]], dtype=object),
        config=TeacherEdgePriorConfig(relief=1.0, teacher_cost_cap=2.0, min_cost=0.0),
    )

    assert adjusted[(0, 1)][1, 0] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"relief": True}, "relief must be finite"),
        ({"relief": np.bool_(True)}, "relief must be finite"),
        (
            {"teacher_cost_cap": -0.1},
            "teacher_cost_cap must be finite and non-negative",
        ),
        ({"non_teacher_penalty": np.nan}, "non_teacher_penalty must be finite"),
        ({"min_cost": np.inf}, "min_cost must be finite"),
        ({"max_gap": 1.5}, "max_gap must be a positive integer"),
        ({"consecutive_only": 1}, "consecutive_only must be a boolean"),
        ({"large_cost": 0.0}, "large_cost must be finite and positive"),
    ],
)
def test_teacher_prior_config_rejects_silent_runtime_knob_coercions(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TeacherEdgePriorConfig(**kwargs)


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
