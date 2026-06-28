from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.dynamic_edge_priors import DynamicEdgePriorConfig


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("session_gap_weight", np.array(True)),
        ("reciprocal_rank_cap", np.array(False)),
        ("edge_quality_bias", np.array(True, dtype=object)),
        ("large_cost", np.array("1.0")),
    ],
)
def test_dynamic_edge_prior_config_rejects_non_numeric_scalar_arrays(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        DynamicEdgePriorConfig(**{field_name: bad_value})
