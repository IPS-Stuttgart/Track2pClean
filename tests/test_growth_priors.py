from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.growth_priors import (
    GrowthPriorConfig,
    fit_affine_growth_transform,
    radial_growth_penalty_matrix,
)


def test_growth_prior_config_normalizes_numeric_controls() -> None:
    config = GrowthPriorConfig(
        affine_weight="0.5",  # type: ignore[arg-type]
        radial_weight=np.float64(0.25),
        displacement_scale="12.0",  # type: ignore[arg-type]
        regularization="1e-4",  # type: ignore[arg-type]
    )

    assert config.affine_weight == pytest.approx(0.5)
    assert config.radial_weight == pytest.approx(0.25)
    assert config.displacement_scale == pytest.approx(12.0)
    assert config.regularization == pytest.approx(1.0e-4)


@pytest.mark.parametrize("field", ["affine_weight", "radial_weight", "regularization"])
@pytest.mark.parametrize(
    "bad_value", [True, np.bool_(False), -1.0, float("nan"), float("inf")]
)
def test_growth_prior_config_rejects_invalid_nonnegative_controls(
    field: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        GrowthPriorConfig(**{field: bad_value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_value",
    [True, np.bool_(False), 0.0, -1.0, float("nan"), float("inf")],
)
def test_growth_prior_config_rejects_invalid_displacement_scale(
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match="displacement_scale"):
        GrowthPriorConfig(displacement_scale=bad_value)  # type: ignore[arg-type]


def test_affine_growth_transform_rejects_invalid_regularization() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = np.asarray([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])

    with pytest.raises(ValueError, match="regularization"):
        fit_affine_growth_transform(source, target, regularization=float("nan"))


def test_radial_growth_penalty_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        radial_growth_penalty_matrix(
            np.asarray([[0.0, 0.0]]),
            np.asarray([[1.0, 1.0]]),
            scale=float("inf"),
        )
