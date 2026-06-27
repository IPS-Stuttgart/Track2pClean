from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.absence_model import absence_cost_vector


def test_absence_cost_vector_rejects_wrong_length_registered_mask() -> None:
    plane = SimpleNamespace(
        n_rois=2,
        cell_probabilities=None,
        traces=np.zeros((2, 1), dtype=float),
        spike_traces=None,
    )

    with pytest.raises(ValueError, match="registered_empty_mask"):
        absence_cost_vector(
            plane,
            registered_empty_mask=np.asarray([True], dtype=bool),
        )
