from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.experiments.calibration_gap_weights import (
    balanced_binary_gap_sample_weights,
)


def _gap_balanced_weights(labels: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    features = np.stack([np.zeros_like(gaps, dtype=float), gaps], axis=-1)
    return balanced_binary_gap_sample_weights(
        features,
        labels,
        ("dummy_cost", "session_gap"),
    )


def _assert_normalized_shape(weights: np.ndarray, labels: np.ndarray) -> None:
    assert weights.shape == labels.shape
    assert np.isclose(float(np.sum(weights)), float(labels.size))


def test_gap_balanced_weights_equalize_label_gap_group_mass():
    labels = np.array([1, 0, 0, 0, 1, 1, 0], dtype=int)
    gaps = np.array([1, 1, 1, 1, 2, 2, 2], dtype=float)

    weights = _gap_balanced_weights(labels, gaps)

    _assert_normalized_shape(weights, labels)
    group_masses = []
    for label in (0, 1):
        for gap in (1.0, 2.0):
            mask = (labels == label) & (gaps == gap)
            group_masses.append(float(np.sum(weights[mask])))
    assert group_masses == pytest.approx([labels.size / 4.0] * 4)


def test_gap_balanced_weights_support_pairwise_feature_tensors():
    labels = np.array([[1, 0], [0, 1]], dtype=int)
    gaps = np.array([[1, 1], [2, 2]], dtype=float)

    weights = _gap_balanced_weights(labels, gaps)

    _assert_normalized_shape(weights, labels)
    assert np.isclose(float(np.sum(weights[(labels == 1) & (gaps == 1)])), 1.0)
    assert np.isclose(float(np.sum(weights[(labels == 0) & (gaps == 1)])), 1.0)
    assert np.isclose(float(np.sum(weights[(labels == 1) & (gaps == 2)])), 1.0)
    assert np.isclose(float(np.sum(weights[(labels == 0) & (gaps == 2)])), 1.0)


def test_gap_balanced_weights_treat_single_feature_name_as_one_name():
    labels = np.array([[1, 0], [0, 1]], dtype=int)
    gaps = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    features = gaps[..., None]

    weights = balanced_binary_gap_sample_weights(features, labels, "session_gap")

    _assert_normalized_shape(weights, labels)
    group_masses = []
    for label in (0, 1):
        for gap in (1.0, 2.0):
            mask = (labels == label) & (gaps == gap)
            group_masses.append(float(np.sum(weights[mask])))
    assert group_masses == pytest.approx([1.0] * 4)


def test_gap_balanced_weights_fall_back_to_binary_balance_when_gap_missing():
    labels = np.array([1, 0, 0, 0], dtype=int)
    features = np.zeros((4, 1), dtype=float)

    weights = balanced_binary_gap_sample_weights(
        features,
        labels,
        ("dummy_cost",),
    )

    assert weights.shape == labels.shape
    assert weights[0] == pytest.approx(2.0)
    assert weights[1:] == pytest.approx([2.0 / 3.0] * 3)


def test_gap_balanced_weights_can_require_gap_feature():
    labels = np.array([1, 0], dtype=int)
    features = np.zeros((2, 1), dtype=float)

    with pytest.raises(ValueError, match="gap feature"):
        balanced_binary_gap_sample_weights(
            features,
            labels,
            ("dummy_cost",),
            missing_gap="raise",
        )
