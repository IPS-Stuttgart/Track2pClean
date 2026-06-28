from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.registration import register_measurement_plane_to_reference


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weighted_centroids": "false"}, "weighted_centroids"),
        ({"allow_reflection": "false"}, "allow_reflection"),
        ({"registration_model": "similarity"}, "registration_model"),
        ({"registration_max_cost": True}, "registration_max_cost"),
        ({"registration_max_cost": -1.0}, "registration_max_cost"),
        ({"registration_max_cost": np.nan}, "registration_max_cost"),
        ({"registration_max_cost": [1.0, 2.0]}, "registration_max_cost"),
        ({"registration_max_iterations": True}, "registration_max_iterations"),
        ({"registration_max_iterations": 0}, "registration_max_iterations"),
        ({"registration_max_iterations": 1.5}, "registration_max_iterations"),
        ({"registration_tolerance": True}, "registration_tolerance"),
        ({"registration_tolerance": -1.0}, "registration_tolerance"),
        ({"registration_tolerance": np.inf}, "registration_tolerance"),
        ({"registration_tolerance": [1.0e-8]}, "registration_tolerance"),
        ({"min_matches": True}, "min_matches"),
        ({"min_matches": 0}, "min_matches"),
        ({"min_matches": 1.5}, "min_matches"),
    ],
)
def test_register_measurement_plane_rejects_invalid_registration_controls(
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        register_measurement_plane_to_reference(object(), object(), **kwargs)
