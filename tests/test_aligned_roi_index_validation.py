from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from bayescatrack.reference import load_aligned_subject_reference  # noqa: E402


class _DummyPlaneData:
    def __init__(self, roi_indices: Any):
        self.n_rois = 2
        self.roi_indices = np.asarray(roi_indices, dtype=int)


class _DummySession:
    def __init__(self, roi_indices: Any):
        self.session_name = "2024-05-01_a"
        self.session_date = date(2024, 5, 1)
        self.plane_data = _DummyPlaneData(roi_indices)


@pytest.mark.parametrize(
    "roi_indices",
    [
        [[0], [1]],
        [[0, 1]],
        np.array([[0], [1]]),
    ],
)
def test_load_aligned_subject_reference_rejects_nested_roi_indices(
    monkeypatch: pytest.MonkeyPatch,
    roi_indices: Any,
):
    monkeypatch.setattr(
        "bayescatrack.reference.load_track2p_subject",
        lambda *args, **kwargs: [_DummySession(roi_indices)],
    )

    with pytest.raises(ValueError, match="plane_data.roi_indices"):
        load_aligned_subject_reference("subject_dir", plane_name="plane0")
