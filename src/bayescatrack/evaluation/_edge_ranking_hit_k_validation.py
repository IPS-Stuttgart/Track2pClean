"""Validation shims for edge-ranking summary cutoff and grouping values."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any, Callable

import numpy as np

_ERROR_MESSAGE = "hit_ks must contain positive integer cutoffs"
_GROUP_KEY_ERROR_MESSAGE = "group_keys must be a sequence of unique, non-empty field names"
_SUMMARY_PATCH_MARKER = "_bayescatrack_edge_ranking_group_key_validation_patch"


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


def _validated_group_keys(group_keys: Any) -> tuple[str, ...]:
    if isinstance(group_keys, str):
        raise ValueError(_GROUP_KEY_ERROR_MESSAGE)
    try:
        raw_keys = tuple(group_keys)
    except TypeError as exc:
        raise ValueError(_GROUP_KEY_ERROR_MESSAGE) from exc

    normalized_keys = tuple(_validated_group_key(key) for key in raw_keys)
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError(_GROUP_KEY_ERROR_MESSAGE)
    return normalized_keys


def _validated_group_key(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(_GROUP_KEY_ERROR_MESSAGE)
    return value


def _function_chain_has_patch(function: Any, marker: str) -> bool:
    seen: set[int] = set()
    current: Any = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, marker, False):
            return True
        seen.add(current_id)
        current = getattr(current, "_bayescatrack_original", None)
    return False


def install_edge_ranking_hit_k_validation(
    edge_ranking_module: Any | None = None,
) -> None:
    """Normalize edge-ranking summary cutoff and grouping controls."""

    if edge_ranking_module is None:
        from . import edge_ranking as edge_ranking_module

    edge_ranking_module._validated_hit_k = (
        _validated_hit_k  # pylint: disable=protected-access
    )

    original: Callable[..., Any] = edge_ranking_module.summarize_edge_ranking_rows
    if _function_chain_has_patch(original, _SUMMARY_PATCH_MARKER):
        return

    @wraps(original)
    def summarize_edge_ranking_rows_with_group_key_validation(
        rows: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        normalized_kwargs = dict(kwargs)
        normalized_kwargs["group_keys"] = _validated_group_keys(
            normalized_kwargs.get("group_keys", edge_ranking_module.DEFAULT_GROUP_KEYS)
        )
        return original(rows, *args, **normalized_kwargs)

    setattr(summarize_edge_ranking_rows_with_group_key_validation, _SUMMARY_PATCH_MARKER, True)
    setattr(
        summarize_edge_ranking_rows_with_group_key_validation,
        "_bayescatrack_original",
        original,
    )
    edge_ranking_module.summarize_edge_ranking_rows = summarize_edge_ranking_rows_with_group_key_validation


__all__ = ["install_edge_ranking_hit_k_validation"]
