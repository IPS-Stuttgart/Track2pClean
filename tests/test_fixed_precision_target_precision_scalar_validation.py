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


@pytest.mark.parametrize("target_precisions", [0.95, object()])
def test_fixed_precision_rejects_non_sequence_target_precisions(
    target_precisions: object,
) -> None:
    predicted, reference = _single_complete_track()

    with pytest.raises(ValueError, match="target_precisions"):
        score_complete_tracks_at_fixed_precision(
            predicted,
            reference,
            target_precisions=target_precisions,
        )


@pytest.mark.parametrize(
    "target_precision",
    [[0.95], np.array([0.95]), np.array([[0.95]]), object()],
)
def test_fixed_precision_rejects_nonscalar_target_precision_entries(
    target_precision: object,
) -> None:
    predicted, reference = _single_complete_track()

    with pytest.raises(ValueError, match="target precisions"):
        score_complete_tracks_at_fixed_precision(
            predicted,
            reference,
            target_precisions=(target_precision,),
        )


def test_fixed_precision_keeps_numeric_target_precision_arrays() -> None:
    predicted, reference = _single_complete_track()

    scores = score_complete_tracks_at_fixed_precision(
        predicted,
        reference,
        target_precisions=np.array([0.95]),
    )

    assert scores["complete_tracks_at_fixed_precision_0_95"] == 1
