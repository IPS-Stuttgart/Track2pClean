"""Strict hit@k cutoff validation for edge-ranking summaries."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_edge_ranking_hit_k_validation_patch"
_HIT_K_ERROR = "hit_ks must contain unique positive integer cutoffs"


def install_edge_ranking_hit_k_validation() -> None:
    """Install idempotent strict validation for edge-ranking hit@k cutoffs."""

    from . import edge_ranking as _edge_ranking  # pylint: disable=import-outside-toplevel

    original: Callable[..., Any] = _edge_ranking.summarize_edge_ranking_rows
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def summarize_edge_ranking_rows_with_hit_k_validation(
        rows: Any,
        *,
        group_keys: Any = _edge_ranking.DEFAULT_GROUP_KEYS,
        hit_ks: Any = _edge_ranking.DEFAULT_HIT_KS,
    ) -> Any:
        return original(
            rows,
            group_keys=group_keys,
            hit_ks=_normalize_hit_ks(hit_ks),
        )

    setattr(summarize_edge_ranking_rows_with_hit_k_validation, _PATCH_MARKER, True)
    setattr(
        summarize_edge_ranking_rows_with_hit_k_validation,
        "_bayescatrack_original",
        original,
    )
    _edge_ranking.summarize_edge_ranking_rows = summarize_edge_ranking_rows_with_hit_k_validation


def _normalize_hit_ks(hit_ks: Any) -> tuple[int, ...]:
    try:
        values = tuple(hit_ks)
    except TypeError as exc:
        raise ValueError(_HIT_K_ERROR) from exc

    if not values:
        raise ValueError(_HIT_K_ERROR)

    normalized = tuple(_normalize_hit_k(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(_HIT_K_ERROR)
    return normalized


def _normalize_hit_k(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_HIT_K_ERROR)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_HIT_K_ERROR)
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(_HIT_K_ERROR) from exc

    integer_value = int(integer_value)
    if integer_value <= 0:
        raise ValueError(_HIT_K_ERROR)
    return integer_value


__all__ = ["install_edge_ranking_hit_k_validation"]
