from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.advanced_roi_components import (
    CandidatePruningConfig,
    candidate_mask_from_cost_matrix,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gate_margin": "0.25"}, "gate_margin must be a finite non-negative value"),
        (
            {"gate_margin": np.asarray([0.25])},
            "gate_margin must be a finite non-negative value",
        ),
        (
            {"large_cost": "1.0"},
            "large_cost must be a finite positive value",
        ),
        (
            {"large_cost": np.asarray([1.0])},
            "large_cost must be a finite positive value",
        ),
    ],
)
def test_candidate_pruning_config_rejects_ambiguous_float_scalars(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CandidatePruningConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gate_margin": "0.25"}, "gate_margin must be a finite non-negative value"),
        (
            {"gate_margin": np.asarray([0.25])},
            "gate_margin must be a finite non-negative value",
        ),
        (
            {"large_cost": "1.0"},
            "large_cost must be a finite positive value",
        ),
        (
            {"large_cost": np.asarray([1.0])},
            "large_cost must be a finite positive value",
        ),
    ],
)
def test_candidate_mask_rejects_ambiguous_float_scalars(kwargs, message):
    with pytest.raises(ValueError, match=message):
        candidate_mask_from_cost_matrix(
            np.zeros((2, 2), dtype=float),
            top_k=1,
            **kwargs,
        )
