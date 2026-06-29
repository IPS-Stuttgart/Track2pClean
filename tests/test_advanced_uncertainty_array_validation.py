from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.advanced_uncertainty import (
    EdgeUncertaintyConfig,
    candidate_mask_from_posteriors,
    posterior_probability_matrix,
)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"temperature": np.asarray([2.0])}, "temperature must be a finite scalar"),
        (
            {"uncertainty_penalty_weight": np.asarray([1.0])},
            "uncertainty_penalty_weight must be a finite scalar",
        ),
        (
            {"min_reliability": np.asarray([0.5])},
            "min_reliability must be a finite scalar",
        ),
    ],
)
def test_uncertainty_config_rejects_singleton_array_runtime_knobs(
    config_kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EdgeUncertaintyConfig(**config_kwargs)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"temperature": "2.0"}, "temperature must be a numeric scalar"),
        ({"max_penalty": b"1000000.0"}, "max_penalty must be a numeric scalar"),
        (
            {"min_reliability": np.asarray("0.5")},
            "min_reliability must be a numeric scalar",
        ),
    ],
)
def test_uncertainty_config_rejects_string_like_runtime_knobs(
    config_kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EdgeUncertaintyConfig(**config_kwargs)


def test_posterior_probabilities_reject_singleton_array_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be a finite scalar"):
        posterior_probability_matrix(
            np.asarray([[0.0, 1.0]], dtype=float),
            temperature=np.asarray([2.0]),
        )


@pytest.mark.parametrize(
    "temperature",
    ["2.0", np.asarray("2.0")],
)
def test_posterior_probabilities_reject_string_like_temperature(
    temperature: object,
) -> None:
    with pytest.raises(ValueError, match="temperature must be a numeric scalar"):
        posterior_probability_matrix(
            np.asarray([[0.0, 1.0]], dtype=float),
            temperature=temperature,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"min_probability": np.asarray([0.5])},
            "min_probability must be a finite scalar",
        ),
        ({"row_top_k": np.asarray([1])}, "row_top_k must be a finite scalar"),
        (
            {"column_top_k": np.asarray([1])},
            "column_top_k must be a finite scalar",
        ),
    ],
)
def test_candidate_mask_rejects_singleton_array_pruning_knobs(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        candidate_mask_from_posteriors(np.asarray([[0.5, 0.25]]), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_probability": "0.5"}, "min_probability must be a numeric scalar"),
        ({"row_top_k": b"1"}, "row_top_k must be a numeric scalar"),
        (
            {"column_top_k": np.asarray("1")},
            "column_top_k must be a numeric scalar",
        ),
    ],
)
def test_candidate_mask_rejects_string_like_pruning_knobs(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        candidate_mask_from_posteriors(np.asarray([[0.5, 0.25]]), **kwargs)
