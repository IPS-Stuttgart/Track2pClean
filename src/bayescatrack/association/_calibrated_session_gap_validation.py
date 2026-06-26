"""Validation patch for calibrated-cost session-gap feature construction.

Calibrated association features treat ``session_gap`` as the discrete distance
between acquisition sessions.  Keep the public component helper from accepting
truthy flags, fractional offsets, or non-finite values that would otherwise be
coerced by ``float(...)`` and written into the feature tensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import calibrated_costs as _calibrated_costs
from ._numeric_validation import positive_integer as _positive_integer

_ORIGINAL_FN_ATTR = "_bayescatrack_original_with_session_gap_component"

if not hasattr(_calibrated_costs, _ORIGINAL_FN_ATTR):
    setattr(
        _calibrated_costs,
        _ORIGINAL_FN_ATTR,
        _calibrated_costs.with_session_gap_component,
    )

_ORIGINAL_WITH_SESSION_GAP_COMPONENT = getattr(_calibrated_costs, _ORIGINAL_FN_ATTR)


def with_session_gap_component(
    pairwise_components: Mapping[str, Any],
    *,
    session_gap: Any,
) -> dict[str, np.ndarray]:
    """Return pairwise components with a validated discrete session-gap plane."""

    return _ORIGINAL_WITH_SESSION_GAP_COMPONENT(
        pairwise_components,
        session_gap=_positive_session_gap(session_gap),
    )


def _positive_session_gap(value: Any) -> int:
    try:
        return _positive_integer(value, name="session_gap")
    except ValueError as exc:
        raise ValueError("session_gap must be positive integer") from exc


_calibrated_costs.with_session_gap_component = with_session_gap_component
