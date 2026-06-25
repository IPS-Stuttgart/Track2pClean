from __future__ import annotations

import numpy as np
from bayescatrack import CalciumPlaneData


def _single_roi_plane(probability: float) -> CalciumPlaneData:
    mask = np.zeros((1, 3, 3), dtype=bool)
    mask[0, 1, 1] = True
    return CalciumPlaneData(
        mask,
        cell_probabilities=np.asarray([probability], dtype=float),
    )


def _cell_probability_components(
    reference_probability: float,
    measurement_probability: float,
) -> dict[str, np.ndarray]:
    reference = _single_roi_plane(reference_probability)
    measurement = _single_roi_plane(measurement_probability)
    _, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=0.0,
        cell_probability_weight=1.0,
        return_components=True,
    )
    return components


def test_pairwise_cell_probability_cost_preserves_valid_probabilities() -> None:
    components = _cell_probability_components(0.8, 0.5)

    np.testing.assert_allclose(
        components["cell_probability_available"],
        np.asarray([[1.0]]),
    )
    np.testing.assert_allclose(
        components["cell_probability_cost"],
        np.asarray([[-0.5 * (np.log(0.8) + np.log(0.5))]]),
    )


def test_pairwise_cell_probability_cost_ignores_above_one_probabilities() -> None:
    components = _cell_probability_components(1.2, 0.8)

    np.testing.assert_allclose(
        components["cell_probability_available"],
        np.asarray([[0.0]]),
    )
    np.testing.assert_allclose(
        components["cell_probability_cost"],
        np.asarray([[0.0]]),
    )
