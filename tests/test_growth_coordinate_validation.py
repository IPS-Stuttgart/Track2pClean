from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.growth_priors import (
    affine_growth_penalty_matrix,
    fit_affine_growth_transform,
    radial_growth_penalty_matrix,
)


def _source_points() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _target_points() -> np.ndarray:
    return _source_points() + np.asarray([1.0, 2.0])


@pytest.mark.parametrize(
    ("field_name", "source_bad_value", "target_bad_value"),
    [
        ("source_points_xy", float("nan"), 0.0),
        ("target_points_xy", 0.0, float("inf")),
    ],
)
def test_affine_growth_transform_rejects_nonfinite_coordinates(
    field_name: str,
    source_bad_value: float,
    target_bad_value: float,
) -> None:
    source = _source_points()
    target = _target_points()
    source[0, 0] = source_bad_value
    target[0, 1] = target_bad_value

    with pytest.raises(ValueError, match=field_name):
        fit_affine_growth_transform(source, target)


def test_affine_growth_penalty_rejects_nonfinite_affine_matrix() -> None:
    affine = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, float("nan")]])

    with pytest.raises(ValueError, match="affine_xy"):
        affine_growth_penalty_matrix(
            _source_points(),
            _target_points(),
            affine,
            scale=1.0,
        )


def test_radial_growth_penalty_rejects_nonfinite_centroids() -> None:
    measurement = _target_points()
    measurement[1, 0] = float("inf")

    with pytest.raises(ValueError, match="measurement_centroids_xy"):
        radial_growth_penalty_matrix(_source_points(), measurement, scale=1.0)


def test_radial_growth_penalty_rejects_nonfinite_center() -> None:
    with pytest.raises(ValueError, match="center_xy"):
        radial_growth_penalty_matrix(
            _source_points(),
            _target_points(),
            center_xy=[0.0, float("nan")],
            scale=1.0,
        )


@pytest.mark.parametrize(
    "center_xy",
    [
        [[0.0, 0.0]],
        [[0.0], [0.0]],
        np.asarray([[0.0, 0.0]]),
        np.asarray([[0.0], [0.0]]),
    ],
)
def test_radial_growth_penalty_rejects_nested_center_vector(center_xy: object) -> None:
    with pytest.raises(ValueError, match="center_xy.*shape"):
        radial_growth_penalty_matrix(
            _source_points(),
            _target_points(),
            center_xy=center_xy,
            scale=1.0,
        )
