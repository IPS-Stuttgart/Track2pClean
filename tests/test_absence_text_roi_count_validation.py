from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.absence_model import absence_cost_vector


def test_absence_cost_vector_rejects_string_roi_count() -> None:
    plane = SimpleNamespace(
        n_rois="2",
        cell_probabilities=None,
        traces=np.zeros((1, 1), dtype=float),
        spike_traces=None,
    )

    with pytest.raises(ValueError, match=r"plane\.n_rois"):
        absence_cost_vector(plane)
