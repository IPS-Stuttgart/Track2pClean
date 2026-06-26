from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayescatrack.experiments import track2p_benchmark as benchmark


def _config(**overrides: object) -> benchmark.Track2pBenchmarkConfig:
    return benchmark.Track2pBenchmarkConfig(
        data=Path("."),
        method="global-assignment",
        **overrides,
    )


@pytest.mark.parametrize(
    "field",
    ("sweep_start_costs", "sweep_end_costs", "sweep_gap_penalties"),
)
@pytest.mark.parametrize(
    "malformed_value",
    (True, False, np.bool_(True), np.array(True, dtype=bool)),
)
def test_assignment_prior_sweeps_reject_boolean_cost_values(
    field: str,
    malformed_value: object,
) -> None:
    config = _config(**{field: (malformed_value,)})

    with pytest.raises(ValueError, match=f"{field} values must be finite numbers"):
        benchmark.assignment_prior_settings_from_config(config)


@pytest.mark.parametrize(
    "malformed_value",
    (True, False, np.bool_(True), np.array(False, dtype=bool)),
)
def test_assignment_prior_sweeps_reject_boolean_threshold_values(
    malformed_value: object,
) -> None:
    config = _config(sweep_cost_thresholds=(malformed_value,))

    with pytest.raises(
        ValueError,
        match="sweep_cost_thresholds values must be finite numbers",
    ):
        benchmark.assignment_prior_settings_from_config(config)


def test_assignment_prior_defaults_reject_boolean_cost_values() -> None:
    config = _config(start_cost=True)

    with pytest.raises(
        ValueError,
        match="sweep_start_costs values must be finite numbers",
    ):
        benchmark.assignment_prior_settings_from_config(config)


def test_assignment_prior_defaults_reject_boolean_threshold_values() -> None:
    config = _config(cost_threshold=np.bool_(False))

    with pytest.raises(
        ValueError,
        match="sweep_cost_thresholds values must be finite numbers",
    ):
        benchmark.assignment_prior_settings_from_config(config)


def test_assignment_prior_sweeps_keep_numeric_strings() -> None:
    config = _config(
        sweep_start_costs="1.5",
        sweep_cost_thresholds="none,2.0",
    )

    settings = benchmark.assignment_prior_settings_from_config(config)

    assert [setting.start_cost for setting in settings] == [1.5, 1.5]
    assert [setting.cost_threshold for setting in settings] == [None, 2.0]
