from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.experiments.track2p_emulation_benchmark import (
    _seed_rois_with_outgoing_links,
    emulate_track2p_tracks,
)


def test_gap_rescue_seed_rois_include_direct_skip_links() -> None:
    links = {
        (0, 1): np.zeros((0, 2), dtype=int),
        (0, 2): np.asarray([[3, 7]], dtype=int),
    }

    seeds = _seed_rois_with_outgoing_links(links, max_gap=2, first_session_size=5)

    np.testing.assert_array_equal(seeds, [3])


def test_gap_rescue_seed_rois_ignore_out_of_range_sources() -> None:
    links = {(0, 2): np.asarray([[8, 1]], dtype=int)}

    seeds = _seed_rois_with_outgoing_links(links, max_gap=2, first_session_size=5)

    np.testing.assert_array_equal(seeds, [])


def test_gap_rescue_rejects_invalid_max_gap() -> None:
    with pytest.raises(ValueError, match="max_gap"):
        emulate_track2p_tracks([], max_gap=0)
