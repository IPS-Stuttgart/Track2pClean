from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.fixed_precision import (
    score_complete_tracks_at_fixed_precision,
)


def test_fixed_precision_rejects_boolean_session_indices() -> None:
    with pytest.raises(ValueError, match="boolean session index"):
        score_complete_tracks_at_fixed_precision(
            np.asarray([[0, 1]], dtype=object),
            np.asarray([[0, 1]], dtype=object),
            session_indices=[True],
        )


def test_fixed_precision_rejects_fractional_session_indices() -> None:
    with pytest.raises(ValueError, match="integer-like"):
        score_complete_tracks_at_fixed_precision(
            np.asarray([[0, 1]], dtype=object),
            np.asarray([[0, 1]], dtype=object),
            session_indices=[1.5],
        )


def test_fixed_precision_rejects_duplicate_session_indices() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        score_complete_tracks_at_fixed_precision(
            np.asarray([[0, 1]], dtype=object),
            np.asarray([[0, 1]], dtype=object),
            session_indices=[0, "0"],
        )


def test_fixed_precision_accepts_integer_like_session_strings() -> None:
    scores = score_complete_tracks_at_fixed_precision(
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=object),
        np.asarray([[0, 1, 2], [3, 4, 99]], dtype=object),
        target_precisions=(0.5,),
        session_indices=["0", "1"],
    )

    assert scores["complete_tracks_at_fixed_precision_0_5"] == 2
    assert scores["complete_track_recall_at_fixed_precision_0_5"] == pytest.approx(1.0)


def test_fixed_precision_excludes_incomplete_predictions() -> None:
    scores = score_complete_tracks_at_fixed_precision(
        np.asarray([[0, 1, 2], [3, -1, 5]], dtype=object),
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=object),
        target_precisions=(0.9,),
        track_scores=(0.4, 0.9),
    )

    assert scores["complete_tracks_at_fixed_precision_0_9"] == 1
    assert scores["complete_track_predictions_at_fixed_precision_0_9"] == 1
    assert scores["complete_track_precision_at_fixed_precision_0_9"] == pytest.approx(
        1.0
    )
    assert scores["complete_track_recall_at_fixed_precision_0_9"] == pytest.approx(0.5)


def test_fixed_precision_empty_operating_point_uses_vacuous_recall() -> None:
    scores = score_complete_tracks_at_fixed_precision(
        np.empty((0, 2), dtype=object),
        np.empty((0, 2), dtype=object),
        target_precisions=(0.95,),
    )

    assert scores["complete_tracks_at_fixed_precision_0_95"] == 0
    assert scores["complete_track_predictions_at_fixed_precision_0_95"] == 0
    assert scores["complete_track_precision_at_fixed_precision_0_95"] == pytest.approx(
        1.0
    )
    assert scores["complete_track_recall_at_fixed_precision_0_95"] == pytest.approx(1.0)
