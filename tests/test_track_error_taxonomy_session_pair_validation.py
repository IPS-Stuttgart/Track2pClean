from __future__ import annotations

import pytest
from bayescatrack.evaluation.track_error_taxonomy import classify_track_errors


def test_classify_track_errors_rejects_fractional_session_pair_indices() -> None:
    with pytest.raises(
        ValueError, match="session_pairs contains non-integer session index"
    ):
        classify_track_errors(
            [[1, 2]],
            [[1, 2]],
            session_pairs=((0.5, 1),),
        )
