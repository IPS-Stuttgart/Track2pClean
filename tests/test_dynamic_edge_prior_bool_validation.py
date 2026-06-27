from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    dynamic_edge_prior_config_from_mapping,
)

_NUMERIC_CONTROL_FIELDS = [
    "session_gap_weight",
    "cell_probability_weight",
    "area_ratio_weight",
    "activity_missing_weight",
    "registration_empty_roi_weight",
    "reciprocal_rank_weight",
    "reciprocal_rank_cap",
    "local_margin_weight",
    "local_margin_target",
    "local_margin_cap",
    "edge_quality_bias",
    "large_cost",
]


@pytest.mark.parametrize("field_name", _NUMERIC_CONTROL_FIELDS)
def test_dynamic_edge_prior_config_rejects_boolean_numeric_controls(field_name):
    with pytest.raises(ValueError, match=f"{field_name}.*boolean"):
        DynamicEdgePriorConfig(**{field_name: True})


def test_dynamic_edge_prior_config_rejects_numpy_bool_controls():
    with pytest.raises(ValueError, match="session_gap_weight.*boolean"):
        DynamicEdgePriorConfig(session_gap_weight=np.bool_(True))


def test_dynamic_edge_prior_mapping_rejects_json_boolean_controls():
    with pytest.raises(ValueError, match="cell_probability_weight.*boolean"):
        dynamic_edge_prior_config_from_mapping({"cell_probability_weight": False})


@pytest.mark.parametrize("field_name", _NUMERIC_CONTROL_FIELDS)
def test_dynamic_edge_prior_config_rejects_text_numeric_controls(field_name):
    with pytest.raises(ValueError, match=f"{field_name}.*text"):
        DynamicEdgePriorConfig(**{field_name: "0.5"})


def test_dynamic_edge_prior_mapping_rejects_json_text_controls():
    with pytest.raises(ValueError, match="cell_probability_weight.*text"):
        dynamic_edge_prior_config_from_mapping({"cell_probability_weight": "0.5"})


def test_dynamic_edge_prior_config_rejects_vector_numeric_controls():
    with pytest.raises(ValueError, match="session_gap_weight.*numeric scalar"):
        DynamicEdgePriorConfig(session_gap_weight=np.asarray([0.5]))
