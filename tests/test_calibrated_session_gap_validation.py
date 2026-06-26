from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.calibrated_costs import (
    pairwise_feature_tensor,
    with_session_gap_component,
)


def _components() -> dict[str, np.ndarray]:
    return {"centroid_distance": np.zeros((1, 2), dtype=float)}


@pytest.mark.parametrize(
    "session_gap",
    [False, True, np.bool_(True), float("nan"), float("inf"), -float("inf")],
)
def test_calibrated_session_gap_rejects_nonfinite_and_boolean_values(
    session_gap: object,
) -> None:
    with pytest.raises(ValueError, match=r"session_gap must.*positive"):
        with_session_gap_component(_components(), session_gap=session_gap)


def test_calibrated_session_gap_accepts_positive_integer_like_values() -> None:
    components = with_session_gap_component(_components(), session_gap=np.float64(2.0))

    np.testing.assert_allclose(
        components["session_gap"],
        np.full((1, 2), 2.0),
    )


def test_calibrated_session_gap_feature_tensor_stays_finite() -> None:
    components = with_session_gap_component(_components(), session_gap=2)

    features = pairwise_feature_tensor(components, feature_names=("session_gap",))

    np.testing.assert_allclose(features[:, :, 0], np.full((1, 2), 2.0))
