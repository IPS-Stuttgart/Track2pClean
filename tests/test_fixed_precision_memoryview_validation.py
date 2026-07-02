from __future__ import annotations

from array import array

import numpy as np
import pytest
from bayescatrack.evaluation.fixed_precision import score_complete_tracks_at_fixed_precision


def test_fixed_precision_rejects_memoryview_session_indices() -> None:
    with pytest.raises(ValueError, match="binary buffer"):
        score_complete_tracks_at_fixed_precision(
            np.asarray([[0, 1]], dtype=object),
            np.asarray([[0, 1]], dtype=object),
            session_indices=memoryview(array("I", [0])),
        )


def test_fixed_precision_rejects_memoryview_target_precisions() -> None:
    with pytest.raises(ValueError, match="binary buffer"):
        score_complete_tracks_at_fixed_precision(
            np.asarray([[0, 1]], dtype=object),
            np.asarray([[0, 1]], dtype=object),
            target_precisions=memoryview(array("d", [0.5])),
        )


def test_fixed_precision_rejects_memoryview_track_scores() -> None:
    with pytest.raises(ValueError, match="binary buffer"):
        score_complete_tracks_at_fixed_precision(
            np.asarray([[0, 1], [2, 3]], dtype=object),
            np.asarray([[0, 1], [2, 3]], dtype=object),
            track_scores=memoryview(array("d", [0.9, 0.8])),
        )
