from __future__ import annotations

import numpy as np
from bayescatrack.association._pyrecest_feature_compat import (
    CalibratedPairwiseAssociationModel,
    NamedPairwiseFeatureSchema,
)


class _TwoDimensionalPredictProbaModel:
    def __init__(self) -> None:
        self.seen_shape: tuple[int, ...] | None = None

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        feature_array = np.asarray(features, dtype=float)
        if feature_array.ndim != 2:
            raise AssertionError("predict_proba received a non-2D feature matrix")
        self.seen_shape = feature_array.shape
        positive_probability = feature_array[:, 0]
        return np.column_stack((1.0 - positive_probability, positive_probability))


def test_predict_proba_model_flattens_pairwise_tensor_and_restores_matrix():
    score = np.array(
        [
            [0.2, 0.7, 0.4],
            [0.9, 0.1, 0.6],
        ],
        dtype=float,
    )
    model = _TwoDimensionalPredictProbaModel()
    wrapped = CalibratedPairwiseAssociationModel(
        model=model,
        schema=NamedPairwiseFeatureSchema(("score",)),
    )

    probabilities = wrapped.pairwise_probability_matrix_from_components(
        {"score": score}
    )

    assert model.seen_shape == (score.size, 1)
    np.testing.assert_allclose(probabilities, score)

    costs = wrapped.pairwise_cost_matrix_from_components({"score": score})
    np.testing.assert_allclose(costs, -np.log(score))
