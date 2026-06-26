"""Cell-probability validation patch for the Track2p/Suite2p bridge."""

from __future__ import annotations

from types import ModuleType

import numpy as np

_CELL_PROBABILITY_PATCH_ATTR = "_bayescatrack_cell_probability_bounds_patch"


def install_cell_probability_cost_patch(bridge_impl: ModuleType) -> None:
    """Install an idempotent bounded-probability cost helper."""

    original = (
        bridge_impl._pairwise_cell_probability_cost
    )  # pylint: disable=protected-access
    if getattr(original, _CELL_PROBABILITY_PATCH_ATTR, False):
        return

    def _pairwise_cell_probability_cost(
        probabilities_self: np.ndarray | None,
        probabilities_other: np.ndarray | None,
        *,
        cost_shape: tuple[int, int],
        similarity_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cell-probability costs, ignoring invalid probability cues."""

        zero_cost = np.zeros(cost_shape, dtype=float)
        zero_available = np.zeros(cost_shape, dtype=float)
        if probabilities_self is None or probabilities_other is None:
            return zero_cost, zero_available

        probabilities_self_array = np.asarray(probabilities_self, dtype=float).reshape(
            -1
        )
        probabilities_other_array = np.asarray(
            probabilities_other, dtype=float
        ).reshape(-1)
        if probabilities_self_array.shape != (cost_shape[0],):
            raise ValueError(
                "cell_probabilities for the reference plane must have shape (n_roi,)"
            )
        if probabilities_other_array.shape != (cost_shape[1],):
            raise ValueError(
                "cell_probabilities for the measurement plane must have shape (n_roi,)"
            )

        valid_self = _is_valid_probability_vector(probabilities_self_array)
        valid_other = _is_valid_probability_vector(probabilities_other_array)
        available = valid_self[:, None] & valid_other[None, :]
        if not np.any(available):
            return zero_cost, zero_available

        clipped_self = np.clip(probabilities_self_array, similarity_epsilon, 1.0)
        clipped_other = np.clip(probabilities_other_array, similarity_epsilon, 1.0)
        raw_cost = -0.5 * (
            np.log(clipped_self[:, None]) + np.log(clipped_other[None, :])
        )

        cost = np.zeros(cost_shape, dtype=float)
        cost[available] = raw_cost[available]
        return cost, available.astype(float)

    setattr(_pairwise_cell_probability_cost, _CELL_PROBABILITY_PATCH_ATTR, True)
    setattr(_pairwise_cell_probability_cost, "_bayescatrack_original", original)
    bridge_impl._pairwise_cell_probability_cost = (
        _pairwise_cell_probability_cost  # pylint: disable=protected-access
    )


def _is_valid_probability_vector(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
