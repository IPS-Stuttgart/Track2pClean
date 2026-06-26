"""Guard sklearn probability adapters against negative-only classifier outputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def install_sklearn_probability_adapter_validation() -> None:
    """Patch sklearn-style pairwise adapters to honor estimator.classes_ explicitly."""

    from bayescatrack.experiments import track2p_configurable_loso_calibration as module

    adapter_cls = module.SklearnPairwiseProbabilityAdapter
    original = adapter_cls.predict_match_probability
    if getattr(original, "_bayescatrack_known_class_probability", False):
        return

    def predict_match_probability(self: Any, features: Any) -> np.ndarray:
        x, shape = module._flatten_feature_array(features)  # pylint: disable=protected-access
        probabilities = np.asarray(self.estimator.predict_proba(x), dtype=float)
        if probabilities.ndim == 2:
            classes = list(getattr(self.estimator, "classes_", ()))
            if 1 in classes:
                positive_index = classes.index(1)
                if positive_index >= probabilities.shape[1]:
                    raise ValueError("predict_proba column count does not match estimator.classes_")
                probabilities = probabilities[:, positive_index]
            elif classes:
                probabilities = np.zeros(probabilities.shape[0], dtype=float)
            else:
                probabilities = probabilities[:, -1]
        return np.asarray(probabilities, dtype=float).reshape(shape)

    predict_match_probability._bayescatrack_known_class_probability = True  # type: ignore[attr-defined]
    predict_match_probability._bayescatrack_original = original  # type: ignore[attr-defined]
    adapter_cls.predict_match_probability = predict_match_probability
