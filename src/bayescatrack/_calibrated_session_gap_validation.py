"""Strict validation for calibrated association session-gap features."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

_PATCH_ATTR = "_bayescatrack_calibrated_session_gap_validation_patch"


def install_calibrated_session_gap_validation() -> None:
    """Install idempotent validation for calibrated-cost ``session_gap`` values."""

    from .association import calibrated_costs  # pylint: disable=import-outside-toplevel

    if getattr(calibrated_costs, _PATCH_ATTR, False):
        return

    original = calibrated_costs.with_session_gap_component

    def with_session_gap_component_with_validation(
        pairwise_components: Mapping[str, Any],
        *,
        session_gap: Any,
    ) -> dict[str, np.ndarray]:
        return original(
            pairwise_components,
            session_gap=_finite_positive_session_gap(session_gap),
        )

    setattr(with_session_gap_component_with_validation, _PATCH_ATTR, True)
    setattr(
        with_session_gap_component_with_validation,
        "_bayescatrack_original",
        original,
    )
    calibrated_costs.with_session_gap_component = with_session_gap_component_with_validation
    setattr(calibrated_costs, _PATCH_ATTR, True)


def _finite_positive_session_gap(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("session_gap must be a finite positive value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("session_gap must be a finite positive value") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("session_gap must be a finite positive value")
    return numeric


__all__ = ["install_calibrated_session_gap_validation"]
