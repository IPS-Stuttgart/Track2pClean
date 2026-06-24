from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    dynamic_edge_prior_config_from_mapping,
)


@pytest.mark.parametrize(
    "field_name",
    [
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
    ],
)
def test_dynamic_edge_prior_config_rejects_boolean_numeric_controls(field_name):
    with pytest.raises(ValueError, match=f"{field_name}.*boolean"):
        DynamicEdgePriorConfig(**{field_name: True})


def test_dynamic_edge_prior_config_rejects_numpy_bool_controls():
    with pytest.raises(ValueError, match="session_gap_weight.*boolean"):
        DynamicEdgePriorConfig(session_gap_weight=np.bool_(True))


def test_dynamic_edge_prior_mapping_rejects_json_boolean_controls():
    with pytest.raises(ValueError, match="cell_probability_weight.*boolean"):
        dynamic_edge_prior_config_from_mapping({"cell_probability_weight": False})
