from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from bayescatrack.association.adaptive_priors import (
    AdaptiveEdgePriorConfig,
    apply_adaptive_edge_priors,
    fit_gap_costs_from_reference,
)


class _ReferenceForGapCosts:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = np.asarray(matrix, dtype=object)
        self.n_sessions = int(self._matrix.shape[1])

    def filtered_indices(self, *, curated_only: bool = False) -> np.ndarray:
        return self._matrix


@pytest.mark.parametrize(
    "field",
    [
        "session_gap_weight",
        "border_proximity_weight",
        "low_cell_probability_weight",
        "mask_fragility_weight",
    ],
)
@pytest.mark.parametrize("bad_value", [True, np.bool_(False), -0.1, np.nan, np.inf])
def test_adaptive_edge_prior_config_rejects_invalid_nonnegative_weights(
    field: str, bad_value: object
) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite and non-negative"):
        AdaptiveEdgePriorConfig(**{field: bad_value})


@pytest.mark.parametrize(
    ("learned_gap_costs", "message"),
    [
        ({0: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({-1: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({1.5: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({np.inf: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({True: 1.0}, "learned_gap_costs key must be a positive integer"),
        ({1: np.nan}, "learned_gap_costs value must be finite and non-negative"),
        ({1: np.inf}, "learned_gap_costs value must be finite and non-negative"),
        ({1: -0.1}, "learned_gap_costs value must be finite and non-negative"),
        ({1: False}, "learned_gap_costs value must be finite and non-negative"),
    ],
)
def test_adaptive_edge_prior_config_rejects_invalid_learned_gap_costs(
    learned_gap_costs: dict[object, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveEdgePriorConfig(learned_gap_costs=learned_gap_costs)


def test_adaptive_edge_prior_config_coerces_valid_learned_gap_costs() -> None:
    config = AdaptiveEdgePriorConfig(learned_gap_costs={"1": "0.25", 2: 0.5})

    assert config.learned_gap_costs == {1: 0.25, 2: 0.5}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"large_cost": np.nan}, "large_cost must be a positive finite value"),
        ({"large_cost": np.inf}, "large_cost must be a positive finite value"),
        ({"large_cost": 0.0}, "large_cost must be a positive finite value"),
        ({"large_cost": True}, "large_cost must be a positive finite value"),
    ],
)
def test_adaptive_edge_prior_config_rejects_invalid_large_cost(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveEdgePriorConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_gap": True}, "max_gap must be a positive integer"),
        ({"max_gap": 0}, "max_gap must be a positive integer"),
        ({"max_gap": 1.5}, "max_gap must be a positive integer"),
        ({"max_gap": np.inf}, "max_gap must be a positive integer"),
        ({"smoothing": True}, "smoothing must be a positive finite value"),
        ({"smoothing": 0.0}, "smoothing must be a positive finite value"),
        ({"smoothing": np.nan}, "smoothing must be a positive finite value"),
        ({"smoothing": np.inf}, "smoothing must be a positive finite value"),
    ],
)
def test_fit_gap_costs_from_reference_rejects_invalid_controls(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    reference = _ReferenceForGapCosts(np.asarray([[0, 0, 0]], dtype=object))

    with pytest.raises(ValueError, match=message):
        fit_gap_costs_from_reference(reference, **kwargs)


def test_fit_gap_costs_from_reference_normalizes_numeric_controls() -> None:
    reference = _ReferenceForGapCosts(
        np.asarray(
            [
                [0, 0, 0],
                [1, None, 1],
            ],
            dtype=object,
        )
    )

    costs = fit_gap_costs_from_reference(
        reference,
        max_gap="2",  # type: ignore[arg-type]
        smoothing="0.5",  # type: ignore[arg-type]
    )

    assert set(costs) == {1, 2}
    assert all(np.isfinite(cost) for cost in costs.values())


def test_adaptive_edge_priors_reject_fractional_session_edge(make_track2p_session):
    masks = np.ones((1, 2, 2), dtype=bool)
    sessions = (
        make_track2p_session("2024-01-01_a", masks),
        make_track2p_session("2024-01-02_a", masks),
    )

    with pytest.raises(ValueError, match="session edge source"):
        apply_adaptive_edge_priors({(0.25, 1): np.zeros((1, 1))}, sessions)


def test_adaptive_edge_priors_reject_boolean_session_edge(make_track2p_session):
    masks = np.ones((1, 2, 2), dtype=bool)
    sessions = (
        make_track2p_session("2024-01-01_a", masks),
        make_track2p_session("2024-01-02_a", masks),
    )

    with pytest.raises(ValueError, match="session edge source"):
        apply_adaptive_edge_priors({(True, 1): np.zeros((1, 1))}, sessions)


def test_adaptive_edge_priors_reject_malformed_edge_key_shape(make_track2p_session):
    masks = np.ones((1, 2, 2), dtype=bool)
    sessions = (
        make_track2p_session("2024-01-01_a", masks),
        make_track2p_session("2024-01-02_a", masks),
    )

    with pytest.raises(ValueError, match="length-2 session-edge pairs"):
        apply_adaptive_edge_priors({(0, 1, 2): np.zeros((1, 1))}, sessions)

    with pytest.raises(ValueError, match="length-2 session-edge pairs"):
        apply_adaptive_edge_priors({"01": np.zeros((1, 1))}, sessions)


@pytest.mark.parametrize("edge", [(0, 0), (1, 0), (0, 2), (2, 3)])
def test_adaptive_edge_priors_reject_invalid_session_edge_when_disabled(
    make_track2p_session,
    edge: tuple[int, int],
) -> None:
    masks = np.ones((1, 2, 2), dtype=bool)
    sessions = (
        make_track2p_session("2024-01-01_a", masks),
        make_track2p_session("2024-01-02_a", masks),
    )

    with pytest.raises(ValueError, match="Invalid session edge"):
        apply_adaptive_edge_priors({edge: np.zeros((1, 1))}, sessions)


def test_adaptive_edge_priors_reject_shape_mismatch_when_disabled(
    make_track2p_session,
) -> None:
    source_masks = np.ones((1, 2, 2), dtype=bool)
    target_masks = np.ones((2, 2, 2), dtype=bool)
    sessions = (
        make_track2p_session("2024-01-01_a", source_masks),
        make_track2p_session("2024-01-02_a", target_masks),
    )

    with pytest.raises(ValueError, match="does not match the loaded session ROI counts"):
        apply_adaptive_edge_priors({(0, 1): np.zeros((1, 1))}, sessions)
