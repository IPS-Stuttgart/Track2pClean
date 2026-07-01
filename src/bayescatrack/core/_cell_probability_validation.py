"""Cell-probability validation patch for the Track2p/Suite2p bridge."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_CELL_PROBABILITY_COST_PATCH_ATTR = "_bayescatrack_cell_probability_bounds_patch"
_CELL_PROBABILITY_INIT_PATCH_ATTR = "_bayescatrack_cell_probability_init_bounds_patch"


def install_cell_probability_cost_patch(bridge_impl: ModuleType) -> None:
    """Install idempotent bounded-probability validation and cost helpers."""

    _install_calcium_plane_cell_probability_validation(bridge_impl.CalciumPlaneData)
    _install_pairwise_cell_probability_cost_validation(bridge_impl)


def _install_calcium_plane_cell_probability_validation(
    calcium_plane_cls: type[Any],
) -> None:
    """Reject invalid cell-probability arrays at plane construction time."""

    original_post_init = calcium_plane_cls.__post_init__
    if _wrapper_chain_has_attr(original_post_init, _CELL_PROBABILITY_INIT_PATCH_ATTR):
        return

    def __post_init__(self: Any) -> None:
        original_post_init(self)
        probabilities = self.cell_probabilities
        if probabilities is None:
            return

        probabilities_array = np.asarray(probabilities, dtype=float)
        if probabilities_array.shape != (self.n_rois,):
            raise ValueError("cell_probabilities must have shape (n_roi,)")
        if not np.all(_is_valid_probability_vector(probabilities_array)):
            raise ValueError(
                "cell_probabilities must be finite probabilities between 0 and 1"
            )
        object.__setattr__(self, "cell_probabilities", probabilities_array)

    setattr(__post_init__, _CELL_PROBABILITY_INIT_PATCH_ATTR, True)
    setattr(__post_init__, "_bayescatrack_original", original_post_init)
    calcium_plane_cls.__post_init__ = __post_init__


def _install_pairwise_cell_probability_cost_validation(bridge_impl: ModuleType) -> None:
    """Reject invalid probability cues before computing pairwise costs."""

    original = (
        bridge_impl._pairwise_cell_probability_cost
    )  # pylint: disable=protected-access
    if _wrapper_chain_has_attr(original, _CELL_PROBABILITY_COST_PATCH_ATTR):
        return

    def _pairwise_cell_probability_cost(
        probabilities_self: np.ndarray | None,
        probabilities_other: np.ndarray | None,
        *,
        cost_shape: tuple[int, int],
        similarity_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cell-probability costs for validated bounded probabilities."""

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
        if not np.all(_is_valid_probability_vector(probabilities_self_array)):
            raise ValueError(
                "cell_probabilities for the reference plane must be finite probabilities between 0 and 1"
            )
        if not np.all(_is_valid_probability_vector(probabilities_other_array)):
            raise ValueError(
                "cell_probabilities for the measurement plane must be finite probabilities between 0 and 1"
            )

        available = np.ones(cost_shape, dtype=bool)
        if not np.any(available):
            return zero_cost, zero_available

        clipped_self = np.clip(probabilities_self_array, similarity_epsilon, 1.0)
        clipped_other = np.clip(probabilities_other_array, similarity_epsilon, 1.0)
        cost = -0.5 * (np.log(clipped_self[:, None]) + np.log(clipped_other[None, :]))
        return cost, available.astype(float)

    setattr(_pairwise_cell_probability_cost, _CELL_PROBABILITY_COST_PATCH_ATTR, True)
    setattr(_pairwise_cell_probability_cost, "_bayescatrack_original", original)
    bridge_impl._pairwise_cell_probability_cost = (
        _pairwise_cell_probability_cost  # pylint: disable=protected-access
    )


def _wrapper_chain_has_attr(function: Any, marker: str) -> bool:
    seen: set[int] = set()
    current = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        seen.add(current_id)
        if getattr(current, marker, False):
            return True
        current = getattr(current, "_bayescatrack_original", None)
    return False


def _is_valid_probability_vector(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
