from __future__ import annotations

import numpy as np

from bayescatrack.association.segmentation_events import event_soft_penalty_matrix


def test_event_soft_penalty_matrix_accepts_mapping_config() -> None:
    components = {
        "overlap_min_fraction": np.asarray([[0.5, 0.0]], dtype=float),
        "weighted_dice_similarity": np.asarray([[0.6, 0.9]], dtype=float),
        "area_ratio_cost": np.asarray([[0.8, 2.0]], dtype=float),
    }

    relief = event_soft_penalty_matrix(
        components,
        config={
            "min_overlap_fraction": 0.4,
            "min_weighted_dice": 0.5,
            "max_area_ratio_cost": 1.0,
        },
    )

    np.testing.assert_allclose(relief, np.asarray([[-0.1375, 0.0]], dtype=float))
