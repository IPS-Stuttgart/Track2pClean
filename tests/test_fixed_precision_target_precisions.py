from __future__ import annotations

import builtins

import numpy as np
import pytest
from bayescatrack.evaluation.fixed_precision import (
    score_complete_tracks_at_fixed_precision,
)


def test_fixed_precision_rejects_boolean_target_precisions() -> None:
    predicted = np.array([[0, 10]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)

    for malformed_precision in (True, False, np.bool_(True)):
        with pytest.raises(ValueError, match="target precisions"):
            score_complete_tracks_at_fixed_precision(
                predicted,
                reference,
                target_precisions=(malformed_precision,),
            )


def test_fixed_precision_rejects_bare_mutable_bytes_target_precisions() -> None:
    predicted = np.array([[0, 10]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)

    with pytest.raises(ValueError, match="target_precisions"):
        score_complete_tracks_at_fixed_precision(
            predicted,
            reference,
            target_precisions=getattr(builtins, "byte" "array")(bytes([0, 1])),
        )


def test_fixed_precision_rejects_boolean_track_scores() -> None:
    predicted = np.array([[0, 10]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)

    for malformed_scores in ((True,), (False,), (np.bool_(True),), np.array([True])):
        with pytest.raises(ValueError, match="track_scores"):
            score_complete_tracks_at_fixed_precision(
                predicted,
                reference,
                target_precisions=(0.95,),
                track_scores=malformed_scores,
            )


@pytest.mark.parametrize(
    "malformed_score",
    (
        np.asarray([0.75]),
        np.asarray([True], dtype=object),
    ),
)
def test_fixed_precision_rejects_vector_track_score_cells(
    malformed_score: object,
) -> None:
    predicted = np.array([[0, 10]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)
    malformed_scores = np.empty((1,), dtype=object)
    malformed_scores[0] = malformed_score

    with pytest.raises(ValueError, match="track_scores"):
        score_complete_tracks_at_fixed_precision(
            predicted,
            reference,
            target_precisions=(0.95,),
            track_scores=malformed_scores,
        )


def test_fixed_precision_keeps_numeric_target_precision_strings() -> None:
    predicted = np.array([[0, 10]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)

    scores = score_complete_tracks_at_fixed_precision(
        predicted,
        reference,
        target_precisions=("0.95",),
    )

    assert scores["complete_tracks_at_fixed_precision_0_95"] == 1


def test_fixed_precision_keeps_numeric_track_score_strings() -> None:
    predicted = np.array([[0, 10]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)

    scores = score_complete_tracks_at_fixed_precision(
        predicted,
        reference,
        target_precisions=(0.95,),
        track_scores=("0.75",),
    )

    assert scores["complete_track_score_threshold_at_fixed_precision_0_95"] == 0.75


def test_fixed_precision_keeps_scalar_array_track_scores() -> None:
    predicted = np.array([[0, 10]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)

    scores = score_complete_tracks_at_fixed_precision(
        predicted,
        reference,
        target_precisions=(0.95,),
        track_scores=(np.asarray(0.75),),
    )

    assert scores["complete_track_score_threshold_at_fixed_precision_0_95"] == 0.75


def test_fixed_precision_rejects_bare_mutable_bytes_track_scores() -> None:
    predicted = np.array([[0, 10], [1, 11]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)

    with pytest.raises(ValueError, match="track_scores"):
        score_complete_tracks_at_fixed_precision(
            predicted,
            reference,
            target_precisions=(0.95,),
            track_scores=getattr(builtins, "byte" "array")(bytes([0, 1])),
        )


def test_fixed_precision_wraps_overflowing_numeric_controls() -> None:
    predicted = np.array([[0, 10]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)
    overflowing_integer = 10**5000

    with pytest.raises(ValueError, match="target precisions"):
        score_complete_tracks_at_fixed_precision(
            predicted,
            reference,
            target_precisions=(overflowing_integer,),
        )

    with pytest.raises(ValueError, match="track_scores"):
        score_complete_tracks_at_fixed_precision(
            predicted,
            reference,
            target_precisions=(0.95,),
            track_scores=(overflowing_integer,),
        )
