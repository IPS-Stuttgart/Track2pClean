from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bayescatrack.experiments.track2p_policy_gap_pruned import (
    seed_rois_with_outgoing_gap_links,
    tracks_from_gap_links,
)


@dataclass(frozen=True)
class _Plane:
    roi_indices: np.ndarray

    @property
    def n_rois(self) -> int:
        return int(self.roi_indices.size)


@dataclass(frozen=True)
class _Session:
    roi_indices: tuple[int, ...]

    @property
    def plane_data(self) -> _Plane:
        return _Plane(np.asarray(self.roi_indices, dtype=int))


def test_seed_rois_with_outgoing_gap_links_includes_skip_link_starters() -> None:
    links_by_gap = {
        (0, 1): np.zeros((0, 2), dtype=int),
        (0, 2): np.asarray([[3, 5], [1, 2]], dtype=int),
    }

    seeds = seed_rois_with_outgoing_gap_links(
        links_by_gap, max_gap=2, first_session_size=4
    )

    np.testing.assert_array_equal(seeds, [1, 3])


def test_tracks_from_gap_links_prefers_consecutive_before_skip_links() -> None:
    sessions = [_Session((10, 11)), _Session((20, 21)), _Session((30, 31))]
    links_by_gap = {
        (0, 1): np.asarray([[0, 1]], dtype=int),
        (0, 2): np.asarray([[0, 0], [1, 1]], dtype=int),
        (1, 1): np.asarray([[1, 1]], dtype=int),
    }

    tracks = tracks_from_gap_links(  # type: ignore[arg-type]
        sessions, links_by_gap, max_gap=2
    )

    np.testing.assert_array_equal(
        tracks,
        [
            [10, 21, 31],
            [11, -1, 31],
        ],
    )


def test_tracks_from_gap_links_uses_skip_when_consecutive_missing() -> None:
    sessions = [_Session((10,)), _Session((20,)), _Session((30,))]
    links_by_gap = {
        (0, 1): np.zeros((0, 2), dtype=int),
        (0, 2): np.asarray([[0, 0]], dtype=int),
        (1, 1): np.zeros((0, 2), dtype=int),
    }

    tracks = tracks_from_gap_links(  # type: ignore[arg-type]
        sessions, links_by_gap, max_gap=2
    )

    np.testing.assert_array_equal(tracks, [[10, -1, 30]])


def test_tracks_from_gap_links_uses_skip_to_avoid_consecutive_dead_end() -> None:
    sessions = [_Session((10,)), _Session((20,)), _Session((30,)), _Session((40,))]
    links_by_gap = {
        (0, 1): np.asarray([[0, 0]], dtype=int),
        (0, 2): np.asarray([[0, 0]], dtype=int),
        (1, 1): np.zeros((0, 2), dtype=int),
        (1, 2): np.zeros((0, 2), dtype=int),
        (2, 1): np.asarray([[0, 0]], dtype=int),
    }

    tracks = tracks_from_gap_links(  # type: ignore[arg-type]
        sessions, links_by_gap, max_gap=2
    )

    np.testing.assert_array_equal(tracks, [[10, -1, 30, 40]])
