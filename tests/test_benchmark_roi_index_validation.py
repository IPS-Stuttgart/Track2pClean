from __future__ import annotations

import numpy as np
from bayescatrack.experiments import track2p_benchmark as benchmark


def test_benchmark_roi_predicate_rejects_fractional_and_boolean_indices():
    valid_values = [0, np.int64(1), 2.0, np.float64(3.0), "4"]
    invalid_values = [
        True,
        np.bool_(False),
        1.25,
        np.float64(2.5),
        np.nan,
        np.inf,
        -1,
        "-1",
        "1.0",
        object(),
    ]

    for value in valid_values:
        assert benchmark._is_valid_roi_index(value)
    for value in invalid_values:
        assert not benchmark._is_valid_roi_index(value)


def test_reference_seed_roi_set_does_not_truncate_corrupt_indices():
    reference_matrix = np.asarray(
        [[0.0], [1.5], [True], [np.float64(2.0)], ["3"], [-1], [np.nan]],
        dtype=object,
    )

    assert benchmark._reference_seed_roi_set(reference_matrix, seed_session=0) == {
        0,
        2,
        3,
    }
