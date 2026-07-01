"""Regression tests for text-like Mahalanobis scalar controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import numpy as np
import pytest
from bayescatrack.core.bridge import CalciumPlaneData

CostMatrixResult: TypeAlias = np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]


def _plane() -> CalciumPlaneData:
    masks = np.zeros((1, 5, 5), dtype=bool)
    masks[0, 1:3, 1:3] = True
    return CalciumPlaneData(roi_masks=masks)


def _mahalanobis_distances(
    reference: CalciumPlaneData,
    measurement: CalciumPlaneData,
    **kwargs: Any,
) -> np.ndarray:
    method = cast(
        Callable[..., np.ndarray],
        getattr(reference, "pairwise_mahalanobis_centroid_distances"),
    )
    return method(measurement, **kwargs)


def _build_pairwise_cost_matrix(
    reference: CalciumPlaneData,
    measurement: CalciumPlaneData,
    **kwargs: Any,
) -> CostMatrixResult:
    method = cast(
        Callable[..., CostMatrixResult],
        getattr(reference, "build_pairwise_cost_matrix"),
    )
    return method(measurement, **kwargs)


@pytest.mark.parametrize(
    "regularization",
    ["0.0", np.str_("0.0")],
)
def test_pairwise_mahalanobis_distances_reject_text_regularization(
    regularization: Any,
) -> None:
    plane = _plane()

    with pytest.raises(
        ValueError,
        match="regularization must be a finite non-negative value",
    ):
        _mahalanobis_distances(plane, plane, regularization=regularization)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mahalanobis_weight": "0.5"},
        {"mahalanobis_weight": np.str_("0.5")},
        {"mahalanobis_regularization": "0.0"},
        {"mahalanobis_regularization": np.str_("0.0")},
    ],
)
def test_pairwise_cost_matrix_rejects_text_mahalanobis_controls(
    kwargs: dict[str, Any],
) -> None:
    plane = _plane()
    name = next(iter(kwargs))

    with pytest.raises(
        ValueError,
        match=f"{name} must be a finite non-negative value",
    ):
        _build_pairwise_cost_matrix(plane, plane, **kwargs)
