"""Score-name validation for missing edge-ranking diagnostic rows.

``missing_reference_edge_rows`` emits one synthetic diagnostic row per absent
manual-GT edge and score.  Without validating ``score_names`` first, a bare
string such as ``"iou"`` is treated as an iterable of characters, and duplicate
or empty names can silently suppress or corrupt missing-edge diagnostics.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_edge_ranking_score_name_validation_patch"
_SCORE_NAME_ERROR = "score_names must contain at least one unique, non-empty string"


def install_edge_ranking_score_name_validation(
    edge_ranking_module: Any | None = None,
) -> None:
    """Install idempotent score-name validation on edge-ranking helpers."""

    if edge_ranking_module is None:
        from . import edge_ranking as edge_ranking_module

    original: Callable[..., Any] = edge_ranking_module.missing_reference_edge_rows
    if _function_chain_has_patch(original, _PATCH_MARKER):
        return

    @wraps(original)
    def missing_reference_edge_rows_with_score_name_validation(
        reference_matches: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        normalized_kwargs = dict(kwargs)
        if "score_names" not in normalized_kwargs:
            return original(reference_matches, *args, **normalized_kwargs)
        normalized_kwargs["score_names"] = _normalize_score_names(
            normalized_kwargs["score_names"]
        )
        return original(reference_matches, *args, **normalized_kwargs)

    setattr(missing_reference_edge_rows_with_score_name_validation, _PATCH_MARKER, True)
    setattr(
        missing_reference_edge_rows_with_score_name_validation,
        "_bayescatrack_original",
        original,
    )
    edge_ranking_module.missing_reference_edge_rows = (
        missing_reference_edge_rows_with_score_name_validation
    )


def _normalize_score_names(score_names: Any) -> tuple[str, ...]:
    if isinstance(score_names, (str, bytes, np.str_, np.bytes_)):
        raise ValueError(_SCORE_NAME_ERROR)
    try:
        raw_names = tuple(score_names)
    except TypeError as exc:
        raise ValueError(_SCORE_NAME_ERROR) from exc

    if not raw_names:
        raise ValueError(_SCORE_NAME_ERROR)
    names = tuple(_normalize_score_name(name) for name in raw_names)
    if len(set(names)) != len(names):
        raise ValueError(_SCORE_NAME_ERROR)
    return names


def _normalize_score_name(name: Any) -> str:
    if isinstance(name, (bytes, np.bytes_)):
        try:
            name = bytes(name).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(_SCORE_NAME_ERROR) from exc
    if not isinstance(name, (str, np.str_)):
        raise ValueError(_SCORE_NAME_ERROR)
    normalized = str(name)
    if not normalized:
        raise ValueError(_SCORE_NAME_ERROR)
    return normalized


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


__all__ = ["install_edge_ranking_score_name_validation"]
