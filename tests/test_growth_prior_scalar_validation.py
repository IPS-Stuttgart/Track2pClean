from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.growth_priors import GrowthPriorConfig


def test_growth_prior_config_rejects_vector_weight_control() -> None:
    with pytest.raises(ValueError, match="affine_weight"):
        GrowthPriorConfig(affine_weight=np.array([0.5]))  # type: ignore[arg-type]
