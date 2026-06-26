from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.advanced_uncertainty import (
    EdgeUncertaintyConfig,
    candidate_mask_from_posteriors,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": np.bool_(True)}, "temperature must be finite"),
        ({"min_reliability": np.array(True)}, "min_reliability must be finite"),
        ({"gated_edge_weight": np.bool_(True)}, "gated_edge_weight must be finite"),
    ],
)
def test_edge_uncertainty_config_rejects_numpy_boolean_scalars(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EdgeUncertaintyConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_probability": np.bool_(True)}, "min_probability must be finite"),
        ({"row_top_k": np.bool_(True)}, "row_top_k must be finite"),
        ({"column_top_k": np.array(True)}, "column_top_k must be finite"),
    ],
)
def test_candidate_mask_from_posteriors_rejects_numpy_boolean_scalars(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        candidate_mask_from_posteriors(np.asarray([[0.5, 0.25]]), **kwargs)
