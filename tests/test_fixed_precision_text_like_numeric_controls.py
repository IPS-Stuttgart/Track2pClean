from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.fixed_precision import (
    score_complete_tracks_at_fixed_precision,
)


def _single_complete_track() -> tuple[np.ndarray, np.ndarray]:
    predicted = np.array([[0, 10]], dtype=object)
    reference = np.array([[0, 10]], dtype=object)
    return predicted, reference


@pytest.mark.parametrize(
    "target_precision",
    (
        "0.95",
        b"0.95",
        bytearray(b"0.95"),
        np.asarray("0.95"),
        np.asarray(b"0.95"),
        np.asarray("0.95", dtype=object),
        np.asarray(b"0.95", dtype=object),
    ),
)
def test_fixed_precision_rejects_text_like_target_precision_entries(
    target_precision: object,
) -> None:
    predicted, reference = _single_complete_track()

    with pytest.raises(ValueError, match="target precisions"):
        score_complete_tracks_at_fixed_precision(
            predicted,
            reference,
            target_precisions=(target_precision,),
        )


@pytest.mark.parametrize(
    "track_score",
    (
        "0.9",
        b"0.9",
        bytearray(b"0.9"),
        np.asarray("0.9"),
        np.asarray(b"0.9"),
        np.asarray("0.9", dtype=object),
        np.asarray(b"0.9", dtype=object),
    ),
)
def test_fixed_precision_rejects_text_like_track_score_entries(
    track_score: object,
) -> None:
    predicted, reference = _single_complete_track()

    with pytest.raises(ValueError, match="track_scores"):
        score_complete_tracks_at_fixed_precision(
            predicted,
            reference,
            track_scores=(track_score,),
        )
