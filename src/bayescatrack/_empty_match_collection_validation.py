"""Support explicit empty match collections during track stitching.

``build_track_rows_from_matches`` accepts pair-array match inputs in addition to
``SessionMatchResult`` objects and mappings.  Non-empty list-of-pair inputs are
normalized through NumPy, but an explicit empty list has shape ``(0,)`` and was
previously rejected as an unsupported representation.  Treating that value as an
empty mapping lets callers represent a consecutive session pair with no accepted
links without manufacturing an empty ``(0, 2)`` array.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_empty_match_collection_patch"


def install_empty_match_collection_validation(_matching_module: Any) -> None:
    """Install idempotent normalization for explicit empty match collections."""

    original_normalize = (
        _matching_module._normalize_match_mapping
    )  # pylint: disable=protected-access
    if getattr(original_normalize, _PATCH_MARKER, False):
        return

    session_match_result_type = getattr(_matching_module, "SessionMatchResult", None)

    @wraps(original_normalize)
    def normalize_match_mapping_with_empty_collection_support(
        match: Any,
    ) -> dict[int, int]:
        if session_match_result_type is not None and isinstance(
            match, session_match_result_type
        ):
            return original_normalize(match)
        if isinstance(match, Mapping):
            return original_normalize(match)
        if isinstance(match, tuple) and len(match) == 2:
            return original_normalize(match)

        try:
            match_array = np.asarray(match)
        except ValueError:
            return original_normalize(match)

        if match_array.size == 0:
            if match_array.ndim == 1 or (
                match_array.ndim == 2 and match_array.shape[1] == 2
            ):
                return {}
            raise TypeError("unsupported match representation")

        return original_normalize(match)

    setattr(normalize_match_mapping_with_empty_collection_support, _PATCH_MARKER, True)
    setattr(
        normalize_match_mapping_with_empty_collection_support,
        "_bayescatrack_original",
        original_normalize,
    )
    _matching_module._normalize_match_mapping = normalize_match_mapping_with_empty_collection_support  # pylint: disable=protected-access


__all__ = ["install_empty_match_collection_validation"]
