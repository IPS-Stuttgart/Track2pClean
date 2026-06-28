from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.growth_priors import (
    GrowthPriorConfig,
    apply_growth_prior_to_costs,
    estimate_growth_from_track_rows,
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


def test_apply_growth_prior_rejects_broadcastable_cost_shape() -> None:
    reference_centroids = np.asarray([[0.0, 0.0]])
    measurement_centroids = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    costs = np.zeros((2, 3), dtype=float)

    with pytest.raises(ValueError, match="cost_matrix shape"):
        apply_growth_prior_to_costs(
            costs,
            reference_centroids,
            measurement_centroids,
            config=GrowthPriorConfig(affine_weight=0.0, radial_weight=1.0),
        )


def test_estimate_growth_from_track_rows_preserves_integer_links() -> None:
    track_rows = np.asarray([[0, 10], [1, 11], [2, 12]])
    position_tables = [
        {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]},
        {10: [1.0, 1.0], 11: [2.0, 1.0], 12: [1.0, 2.0]},
    ]

    affine = estimate_growth_from_track_rows(
        track_rows,
        position_tables,
        config=GrowthPriorConfig(regularization=0.0),
    )

    np.testing.assert_allclose(affine, np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]))


@pytest.mark.parametrize(
    "bad_value",
    [True, np.bool_(False), 1.25, float("nan"), float("inf"), "2"],
)
def test_estimate_growth_from_track_rows_rejects_malformed_row_entries(
    bad_value: object,
) -> None:
    track_rows = np.asarray([[0, 10], [1, 11], [2, 12]], dtype=object)
    track_rows[0, 0] = bad_value
    position_tables = [
        {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]},
        {10: [1.0, 1.0], 11: [2.0, 1.0], 12: [1.0, 2.0]},
    ]

    with pytest.raises(ValueError, match="track_rows"):
        estimate_growth_from_track_rows(track_rows, position_tables)


@pytest.mark.parametrize("bad_session", [True, np.bool_(False), 1.5, float("nan"), "0"])
def test_estimate_growth_from_track_rows_rejects_malformed_session_columns(
    bad_session: object,
) -> None:
    track_rows = np.asarray([[0, 10], [1, 11], [2, 12]])
    position_tables = [
        {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]},
        {10: [1.0, 1.0], 11: [2.0, 1.0], 12: [1.0, 2.0]},
    ]

    with pytest.raises(ValueError, match="source_session"):
        estimate_growth_from_track_rows(
            track_rows,
            position_tables,
            source_session=bad_session,  # type: ignore[arg-type]
        )
