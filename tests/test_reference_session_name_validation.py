from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.reference import Track2pReference


def test_track2p_reference_rejects_duplicate_session_names() -> None:
    with pytest.raises(ValueError, match="unique session names"):
        Track2pReference(
            session_names=("day0", "day0"),
            suite2p_indices=np.array([[0, 1]], dtype=object),
            source="unit_test",
        )


def test_track2p_reference_accepts_unique_session_names() -> None:
    reference = Track2pReference(
        session_names=("day0", "day1"),
        suite2p_indices=np.array([[0, 1]], dtype=object),
        source="unit_test",
    )

    assert reference.session_names == ("day0", "day1")
