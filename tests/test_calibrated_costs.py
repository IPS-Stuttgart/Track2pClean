from __future__ import annotations

import numpy as np
from bayescatrack.association.calibrated_costs import (
    supervised_pairwise_mask_from_reference,
)
from bayescatrack.reference import Track2pReference


def test_supervised_pairwise_mask_from_reference_uses_only_annotated_endpoints():
    reference = Track2pReference(
        session_names=("s0", "s1"),
        suite2p_indices=np.array(
            [
                [0, 10],
                [2, None],
                [None, 12],
            ],
            dtype=object,
        ),
    )

    supervised = supervised_pairwise_mask_from_reference(
        reference,
        0,
        1,
        reference_roi_indices=np.array([0, 1, 2], dtype=int),
        measurement_roi_indices=np.array([10, 11, 12], dtype=int),
    )

    expected = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(supervised, expected)
