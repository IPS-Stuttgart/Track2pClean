from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.absence_model import absence_summary


def _plane(n_rois: int) -> SimpleNamespace:
    return SimpleNamespace(
        n_rois=n_rois,
        cell_probabilities=None,
        traces=np.zeros((n_rois, 1), dtype=float),
        spike_traces=None,
    )


def test_absence_summary_rejects_matrix_shaped_explicit_costs() -> None:
    with pytest.raises(ValueError, match="absence cost vectors"):
        absence_summary(_plane(2), costs=np.asarray([[1.0], [2.0]], dtype=float))


def test_absence_summary_rejects_invalid_explicit_cost_values() -> None:
    with pytest.raises(ValueError, match="costs"):
        absence_summary(_plane(2), costs=np.asarray([1.0, np.nan], dtype=float))
    with pytest.raises(ValueError, match="costs"):
        absence_summary(_plane(2), costs=np.asarray([1.0, -0.1], dtype=float))


def test_absence_summary_reports_valid_explicit_cost_vector() -> None:
    summary = absence_summary(_plane(2), costs=np.asarray([1.0, 3.0], dtype=float))

    assert summary == {
        "n_rois": 2,
        "mean_absence_cost": 2.0,
        "median_absence_cost": 2.0,
        "min_absence_cost": 1.0,
        "max_absence_cost": 3.0,
    }
