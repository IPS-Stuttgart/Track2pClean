from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from bayescatrack.reference import Track2pReference  # noqa: E402


def test_to_session_track_labels_rejects_non_negative_fill_value():
    reference = Track2pReference(
        session_names=("day0",),
        suite2p_indices=np.array([[0]], dtype=object),
        source="unit_test",
    )

    with pytest.raises(ValueError, match="fill_value must be negative"):
        reference.to_session_track_labels(fill_value=0)
