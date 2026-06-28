"""Validation shims for edge-ranking summary cutoff values."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

_ERROR_MESSAGE = "hit_ks must contain positive integer cutoffs"


def _validated_hit_k(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_ERROR_MESSAGE)
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_ERROR_MESSAGE)
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(_ERROR_MESSAGE) from exc
    integer_value = int(integer_value)
    if integer_value <= 0:
        raise ValueError(_ERROR_MESSAGE)
    return integer_value


def install_edge_ranking_hit_k_validation(edge_ranking_module: Any | None = None) -> None:
    """Normalize exotic ``__index__`` failures for ``hit_ks`` cutoffs."""

    if edge_ranking_module is None:
        from . import edge_ranking as edge_ranking_module

    edge_ranking_module._validated_hit_k = _validated_hit_k  # pylint: disable=protected-access


__all__ = ["install_edge_ranking_hit_k_validation"]
