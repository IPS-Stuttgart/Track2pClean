from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.reference import Track2pReference


def test_reference_rejects_blank_session_name_entry() -> None:
    with pytest.raises(ValueError, match="non-empty string session names"):
        Track2pReference(
            session_names=("", "day1"),
            suite2p_indices=np.array([[0, 1]], dtype=object),
            source="unit_test",
        )


def test_reference_rejects_numeric_session_name_entry() -> None:
    with pytest.raises(ValueError, match="non-empty string session names"):
        Track2pReference(
            session_names=(0, "day1"),  # type: ignore[arg-type]
            suite2p_indices=np.array([[0, 1]], dtype=object),
            source="unit_test",
        )
