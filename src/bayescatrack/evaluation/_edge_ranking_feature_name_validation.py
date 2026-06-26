"""Feature-name validation for edge-ranking score tensor conversion.

``score_matrices_from_feature_tensor`` maps the last tensor axis to a dictionary
of named score matrices.  Without validating the names first, a bare string such
as ``"iou"`` is treated as an iterable of characters and duplicate names
silently overwrite earlier planes.  Both cases corrupt the feature-to-score
mapping before ranking diagnostics are computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_edge_ranking_feature_name_validation_patch"
_FEATURE_NAME_ERROR = "feature_names must be a sequence of unique, non-empty strings"


def install_edge_ranking_feature_name_validation() -> None:
    """Install idempotent feature-name validation on edge-ranking helpers."""

    from . import edge_ranking as _edge_ranking  # pylint: disable=import-outside-toplevel

    original: Callable[..., Any] = _edge_ranking.score_matrices_from_feature_tensor
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def score_matrices_from_feature_tensor_with_feature_name_validation(
        features: Any,
        feature_names: Sequence[str],
    ) -> dict[str, np.ndarray]:
        return original(features, _normalize_feature_names(feature_names))

    setattr(
        score_matrices_from_feature_tensor_with_feature_name_validation,
        _PATCH_MARKER,
        True,
    )
    setattr(
        score_matrices_from_feature_tensor_with_feature_name_validation,
        "_bayescatrack_original",
        original,
    )
    _edge_ranking.score_matrices_from_feature_tensor = score_matrices_from_feature_tensor_with_feature_name_validation


def _normalize_feature_names(feature_names: Any) -> tuple[str, ...]:
    if isinstance(feature_names, (str, bytes, np.str_, np.bytes_)):
        raise ValueError(_FEATURE_NAME_ERROR)
    try:
        raw_names = tuple(feature_names)
    except TypeError as exc:
        raise ValueError(_FEATURE_NAME_ERROR) from exc

    names = tuple(_normalize_feature_name(name) for name in raw_names)
    if len(set(names)) != len(names):
        raise ValueError(_FEATURE_NAME_ERROR)
    return names


def _normalize_feature_name(name: Any) -> str:
    if isinstance(name, bytes):
        try:
            name = name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(_FEATURE_NAME_ERROR) from exc
    if not isinstance(name, (str, np.str_)):
        raise ValueError(_FEATURE_NAME_ERROR)
    normalized = str(name)
    if not normalized:
        raise ValueError(_FEATURE_NAME_ERROR)
    return normalized


__all__ = ["install_edge_ranking_feature_name_validation"]
