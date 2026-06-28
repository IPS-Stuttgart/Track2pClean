from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.soft_overlap_costs import registered_soft_iou_cost_kwargs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"similarity_epsilon": np.array(1 == 1)},
        {"soft_iou_radius": np.array(2 > 1)},
        {"distance_transform_overlap_radius": np.array(1 > 2)},
        {"distance_transform_overlap_weight": np.array(3 == 3, dtype=object)},
        {"distance_transform_overlap_scale": np.array(3 < 2, dtype=object)},
    ],
)
def test_registered_soft_iou_rejects_logical_numpy_scalar_controls(kwargs):
    control_name = next(iter(kwargs))

    with pytest.raises(ValueError, match=control_name):
        registered_soft_iou_cost_kwargs(**kwargs)
