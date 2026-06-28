from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.experiments import track2p_policy_tracklet_graph_mht as tracklet_graph


def test_tracklet_graph_cached_roi_mask_is_read_only() -> None:
    feature_cache = SimpleNamespace()
    masks = np.zeros((1, 3, 3), dtype=float)
    masks[0, 1, 1] = 1.0

    cached = tracklet_graph._cached_roi_mask(
        feature_cache,
        masks,
        cache_key=("session", 0, 0),
        position=0,
    )

    assert cached.dtype == np.bool_
    assert not cached.flags.writeable
    with pytest.raises(ValueError):
        cached[1, 1] = False

    cached_again = tracklet_graph._cached_roi_mask(
        feature_cache,
        np.zeros_like(masks),
        cache_key=("session", 0, 0),
        position=0,
    )
    assert cached_again is cached
    assert bool(cached_again[1, 1])


def test_tracklet_graph_mask_cache_hardens_existing_mutable_entries() -> None:
    feature_cache = SimpleNamespace()
    mutable_cached = np.ones((2, 2), dtype=bool)
    feature_cache._tracklet_graph_roi_mask_cache = {("stale",): mutable_cached}

    cached = tracklet_graph._cached_roi_mask(
        feature_cache,
        np.zeros((1, 2, 2), dtype=bool),
        cache_key=("stale",),
        position=0,
    )

    assert cached is not mutable_cached
    assert not cached.flags.writeable
    with pytest.raises(ValueError):
        cached[0, 0] = False
    assert bool(feature_cache._tracklet_graph_roi_mask_cache[("stale",)][0, 0])
