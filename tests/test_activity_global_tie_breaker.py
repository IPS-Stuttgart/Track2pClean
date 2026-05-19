from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.activity_tie_breaker import (
    activity_tie_breaker_cost_matrix,
)
from bayescatrack.association.pyrecest_global_assignment import (
    build_registered_pairwise_costs,
    solve_global_assignment_for_sessions,
)
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _activity_tie_breaker_metadata,
    _config_from_args,
    _configured_variant_name,
    build_arg_parser,
)


def test_activity_tie_breaker_cost_matrix_scales_selected_component() -> None:
    pairwise_components = {
        "activity_tiebreaker_cost": np.array([[0.0, 0.5], [1.0, np.nan]], dtype=float)
    }

    tie_breaker_cost = activity_tie_breaker_cost_matrix(pairwise_components, weight=0.1)

    npt.assert_allclose(tie_breaker_cost, np.array([[0.0, 0.05], [0.1, 0.05]]))


def test_activity_tie_breaker_cost_matrix_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        activity_tie_breaker_cost_matrix(
            {"activity_tiebreaker_cost": np.zeros((1, 1))}, weight=-1.0
        )
    with pytest.raises(KeyError, match="missing"):
        activity_tie_breaker_cost_matrix({}, component_name="missing")
    with pytest.raises(ValueError, match="two-dimensional"):
        activity_tie_breaker_cost_matrix(
            {"activity_tiebreaker_cost": np.zeros((1, 1, 1))}
        )


def test_global_assignment_exposes_activity_tie_breaker_parameters() -> None:
    for function in (
        build_registered_pairwise_costs,
        solve_global_assignment_for_sessions,
    ):
        signature = inspect.signature(function)
        assert "activity_tie_breaker_weight" in signature.parameters
        assert "activity_tie_breaker_component" in signature.parameters
        assert "activity_trace_source" in signature.parameters
        assert "activity_event_threshold" in signature.parameters
        assert signature.parameters["activity_tie_breaker_weight"].default == 0.0
        assert (
            signature.parameters["activity_tie_breaker_component"].default
            == "activity_tiebreaker_cost"
        )


def test_track2p_benchmark_cli_exposes_activity_tie_breaker_options() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--data",
            "dummy",
            "--method",
            "global-assignment",
            "--activity-tie-breaker-weight",
            "0.03",
            "--activity-tie-breaker-component",
            "spike_similarity_cost",
            "--activity-trace-source",
            "spike_traces",
            "--activity-event-threshold",
            "0.2",
        ]
    )

    config = _config_from_args(args)

    assert config.activity_tie_breaker_weight == pytest.approx(0.03)
    assert config.activity_tie_breaker_component == "spike_similarity_cost"
    assert config.activity_trace_source == "spike_traces"
    assert config.activity_event_threshold == pytest.approx(0.2)


def test_track2p_benchmark_rejects_negative_activity_weight() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--data",
            "dummy",
            "--method",
            "global-assignment",
            "--activity-tie-breaker-weight",
            "-0.1",
        ]
    )

    with pytest.raises(ValueError, match="activity-tie-breaker-weight"):
        _config_from_args(args)


def test_activity_tie_breaker_updates_variant_and_metadata() -> None:
    config = Track2pBenchmarkConfig(
        data=Path("dummy"),
        method="global-assignment",
        activity_tie_breaker_weight=0.03,
        activity_tie_breaker_component="spike_similarity_cost",
        activity_trace_source="spike_traces",
        activity_event_threshold=0.2,
    )

    assert _configured_variant_name(config) == (
        "Same costs + global assignment + activity tie-breaker 0.03 "
        "(spike_similarity_cost)"
    )
    assert _activity_tie_breaker_metadata(config) == {
        "activity_tie_breaker_weight": 0.03,
        "activity_tie_breaker_component": "spike_similarity_cost",
        "activity_trace_source": "spike_traces",
        "activity_event_threshold": 0.2,
    }
