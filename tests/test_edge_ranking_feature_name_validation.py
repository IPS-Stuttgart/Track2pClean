from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.evaluation.edge_ranking import score_matrices_from_feature_tensor


def test_score_matrices_from_feature_tensor_rejects_bare_string_feature_names() -> None:
    features = np.zeros((1, 1, 3), dtype=float)

    with pytest.raises(ValueError, match="feature_names"):
        score_matrices_from_feature_tensor(features, "iou")


def test_score_matrices_from_feature_tensor_rejects_duplicate_feature_names() -> None:
    features = np.zeros((1, 1, 2), dtype=float)

    with pytest.raises(ValueError, match="unique"):
        score_matrices_from_feature_tensor(features, ("cost", "cost"))


def test_score_matrices_from_feature_tensor_preserves_unique_feature_mapping() -> None:
    features = np.array([[[1.0, 2.0]]], dtype=float)

    matrices = score_matrices_from_feature_tensor(features, ("cost", "iou"))

    assert tuple(matrices) == ("cost", "iou")
    assert matrices["cost"].shape == (1, 1)
    assert matrices["cost"][0, 0] == 1.0
    assert matrices["iou"][0, 0] == 2.0
