from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import _suite2p_validation
from bayescatrack.core import _loader_validation

_BOOL_CONTROL_NAMES = tuple(_loader_validation._SUITE2P_BOOL_CONTROL_DEFAULTS)


def test_loader_strict_bool_accepts_numpy_bool_scalars() -> None:
    assert _loader_validation._strict_bool(np.bool_(True), name="include_behavior") is True
    assert _loader_validation._strict_bool(np.bool_(False), name="include_behavior") is False


def test_suite2p_stat_validation_strict_bool_accepts_numpy_bool_scalars() -> None:
    assert _suite2p_validation._strict_bool(np.bool_(True), name="include_non_cells") is True
    assert _suite2p_validation._strict_bool(np.bool_(False), name="include_non_cells") is False


def test_suite2p_trace_flag_validation_accepts_numpy_bool_scalars() -> None:
    assert _suite2p_validation._strict_python_bool(np.bool_(True), name="load_traces") is True
    assert _suite2p_validation._strict_python_bool(np.bool_(False), name="load_traces") is False


def test_suite2p_loader_controls_normalize_numpy_bool_scalars() -> None:
    controls = _loader_validation._validate_suite2p_loader_controls(
        {
            "include_non_cells": np.bool_(True),
            "weighted_masks": np.bool_(False),
            "exclude_overlapping_pixels": np.bool_(True),
            "load_traces": np.bool_(False),
            "load_spike_traces": np.bool_(True),
            "load_neuropil_traces": np.bool_(True),
            "cell_probability_threshold": 0.25,
        }
    )

    assert controls == {
        "include_non_cells": True,
        "weighted_masks": False,
        "exclude_overlapping_pixels": True,
        "load_traces": False,
        "load_spike_traces": True,
        "load_neuropil_traces": True,
        "cell_probability_threshold": 0.25,
    }
    for name in _BOOL_CONTROL_NAMES:
        assert type(controls[name]) is bool


@pytest.mark.parametrize("bad_value", [1, 0, "true", None, np.array(True)])
def test_loader_strict_bool_still_rejects_ambiguous_non_bool_values(bad_value: object) -> None:
    with pytest.raises(ValueError, match="flag must be a boolean"):
        _loader_validation._strict_bool(bad_value, name="flag")


@pytest.mark.parametrize("bad_value", [1, 0, "true", None, np.array(True)])
def test_suite2p_trace_flag_validation_rejects_ambiguous_non_bool_values(
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match="load_traces must be a boolean"):
        _suite2p_validation._strict_python_bool(bad_value, name="load_traces")
