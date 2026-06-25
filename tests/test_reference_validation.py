from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.reference import (
    Track2pReference,
    score_complete_tracks_against_reference,
)


def _reference(*, curated_mask=None) -> Track2pReference:
    return Track2pReference(
        session_names=("day0", "day1"),
        suite2p_indices=np.array([[0, 10], [1, 11]], dtype=object),
        curated_mask=curated_mask,
        source="unit_test",
    )


def test_track2p_reference_parses_explicit_curated_mask_strings() -> None:
    reference = _reference(curated_mask=np.array(["false", "true"], dtype=object))

    npt.assert_array_equal(reference.curated_mask, np.array([False, True]))
    npt.assert_array_equal(
        reference.filtered_indices(curated_only=True), np.array([[1, 11]], dtype=object)
    )


@pytest.mark.parametrize(
    "curated_mask",
    [
        np.array([True, 2], dtype=object),
        np.array([True, np.nan], dtype=object),
        np.array(["true", "maybe"], dtype=object),
    ],
)
def test_track2p_reference_rejects_ambiguous_curated_mask_values(curated_mask) -> None:
    with pytest.raises(ValueError, match="curated_mask"):
        _reference(curated_mask=curated_mask)


@pytest.mark.parametrize("session_indices", [(0, True), (0, 1.5), "01"])
def test_track2p_reference_rejects_malformed_session_indices(session_indices) -> None:
    reference = _reference(curated_mask=np.array([True, True]))

    with pytest.raises(ValueError, match="session_indices"):
        reference.complete_tracks(session_indices=session_indices)


def test_track2p_reference_rejects_boolean_direct_session_indices() -> None:
    reference = _reference(curated_mask=np.array([True, True]))

    with pytest.raises(ValueError, match="session index"):
        reference.pairwise_matches(True, 1)

    with pytest.raises(ValueError, match="session index"):
        score_complete_tracks_against_reference(
            np.array([[0, 10], [1, 11]], dtype=object),
            reference,
            seed_session=True,
        )
