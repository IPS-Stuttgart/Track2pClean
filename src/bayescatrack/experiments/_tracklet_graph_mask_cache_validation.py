"""Protect tracklet-graph ROI-mask cache entries from caller mutation.

The tracklet graph scorer reuses one cached support mask per ROI boundary.  Returning
that cached array as a normal mutable ``ndarray`` lets any diagnostic or internal
caller mutate the cache through an alias, which can silently change subsequent edge
features for the same ROI.  This module makes cached masks immutable at the cache
boundary.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_tracklet_graph_mask_cache_validation_patch"
_CACHE_ATTRIBUTE = "_tracklet_graph_roi_mask_cache"


def install_tracklet_graph_mask_cache_validation() -> None:
    """Install an idempotent read-only guard for tracklet-graph cached ROI masks."""

    from . import track2p_policy_tracklet_graph_mht as _tracklet_graph  # pylint: disable=import-outside-toplevel

    original_cached_roi_mask = _tracklet_graph._cached_roi_mask
    if getattr(original_cached_roi_mask, _PATCH_MARKER, False):
        return

    @wraps(original_cached_roi_mask)
    def _cached_roi_mask_read_only(
        feature_cache: Any,
        masks: Any,
        *,
        cache_key: tuple[Any, ...],
        position: int,
    ) -> np.ndarray:
        mask_cache = getattr(feature_cache, _CACHE_ATTRIBUTE, None)
        if mask_cache is None:
            mask_cache = {}
            setattr(feature_cache, _CACHE_ATTRIBUTE, mask_cache)

        cached = mask_cache.get(cache_key)
        if cached is None:
            cached = np.array(np.asarray(masks[int(position)]) > 0, dtype=bool, copy=True)
            cached.setflags(write=False)
            mask_cache[cache_key] = cached
            return cached

        if not isinstance(cached, np.ndarray) or cached.dtype != np.bool_ or cached.flags.writeable:
            cached = np.array(cached, dtype=bool, copy=True)
            cached.setflags(write=False)
            mask_cache[cache_key] = cached
        return cached

    setattr(_cached_roi_mask_read_only, _PATCH_MARKER, True)
    setattr(_cached_roi_mask_read_only, "_bayescatrack_original", original_cached_roi_mask)
    _tracklet_graph._cached_roi_mask = _cached_roi_mask_read_only


__all__ = ["install_tracklet_graph_mask_cache_validation"]
