from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.calibration_metrics import brier_score


def test_brier_score_matches_mean_squared_probability_error():
    probabilities = np.array([0.0, 0.25, 0.75, 1.0])
    labels = np.array([0, 0, 1, 1])

    assert brier_score(probabilities, labels) == pytest.approx(0.03125)


def test_brier_score_supports_sample_weights():
    probabilities = np.array([0.0, 1.0, 0.5])
    labels = np.array([0, 0, 1])
    weights = np.array([1.0, 3.0, 1.0])

    assert brier_score(probabilities, labels, sample_weight=weights) == pytest.approx(
        0.65
    )


def test_brier_score_accepts_boolean_labels_as_binary_targets():
    assert brier_score([0.25, 0.75], [False, True]) == pytest.approx(0.0625)


@pytest.mark.parametrize(
    "probabilities",
    [
        [True],
        [np.bool_(False)],
        np.asarray([True], dtype=bool),
        np.asarray([np.bool_(False)], dtype=object),
    ],
)
def test_brier_score_rejects_boolean_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilities must be numeric, not boolean"):
        brier_score(probabilities, [1])


@pytest.mark.parametrize(
    "sample_weight",
    [
        [True],
        [np.bool_(False)],
        np.asarray([True], dtype=bool),
        np.asarray([np.bool_(False)], dtype=object),
    ],
)
def test_brier_score_rejects_boolean_sample_weights(sample_weight):
    with pytest.raises(ValueError, match="sample_weight must be numeric, not boolean"):
        brier_score([0.5], [1], sample_weight=sample_weight)


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        ([0.2], [0, 1], "same flattened shape"),
        ([], [], "At least one"),
        ([1.2], [1], "lie in"),
        ([0.5], [2], "binary"),
    ],
)
def test_brier_score_rejects_invalid_inputs(probabilities, labels, message):
    with pytest.raises(ValueError, match=message):
        brier_score(probabilities, labels)
