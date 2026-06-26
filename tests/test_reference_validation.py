from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.evaluation.track2p_metrics import score_track_matrix_against_reference
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


def test_track2p_reference_accepts_numpy_boolean_curated_only() -> None:
    reference = _reference(curated_mask=np.array([True, False], dtype=bool))

    npt.assert_array_equal(
        reference.filtered_indices(curated_only=np.bool_(True)),
        np.array([[0, 10]], dtype=object),
    )
    scores = score_track_matrix_against_reference(
        np.array([[0, 10], [1, 11]], dtype=object),
        reference,
        curated_only=np.bool_(True),
    )
    assert scores["reference_complete_tracks"] == 1
    assert scores["complete_track_precision"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "curated_only",
    ["false", "true", 1, 0, np.array(True, dtype=bool)],
)
def test_track2p_reference_rejects_malformed_curated_only(curated_only) -> None:
    reference = _reference(curated_mask=np.array([True, False], dtype=bool))

    with pytest.raises(ValueError, match="curated_only"):
        reference.filtered_indices(curated_only=curated_only)

    with pytest.raises(ValueError, match="curated_only"):
        score_track_matrix_against_reference(
            np.array([[0, 10], [1, 11]], dtype=object),
            reference,
            curated_only=curated_only,
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


def test_track2p_reference_preserves_explicit_missing_roi_values() -> None:
    reference = Track2pReference(
        session_names=("day0", "day1"),
        suite2p_indices=np.array(
            [
                [0, ""],
                ["-1", None],
                [np.nan, "null"],
            ],
            dtype=object,
        ),
        source="unit_test",
    )

    npt.assert_array_equal(
        reference.present_mask(),
        np.array(
            [
                [True, False],
                [False, False],
                [False, False],
            ]
        ),
    )


@pytest.mark.parametrize("bad_value", ["typo", b"typo", object()])
def test_track2p_reference_rejects_unrecognized_missing_roi_tokens(bad_value) -> None:
    with pytest.raises(ValueError, match="integer-like or an explicit missing value"):
        Track2pReference(
            session_names=("day0",),
            suite2p_indices=np.array([[bad_value]], dtype=object),
            source="unit_test",
        )


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
