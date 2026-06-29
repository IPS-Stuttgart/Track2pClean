from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.absence_model import (
    apply_absence_adjustment,
    gap_penalty_matrix,
)


class _OverflowingFloat:
    def __float__(self) -> float:
        raise OverflowError("numeric adapter overflow")


def _plane(n_rois: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        n_rois=n_rois,
        cell_probabilities=None,
        traces=np.zeros((n_rois, 1), dtype=float),
        spike_traces=None,
    )


def test_gap_penalty_matrix_rejects_overflowing_session_gap() -> None:
    with pytest.raises(ValueError, match="session_gap"):
        gap_penalty_matrix(_plane(), _plane(), session_gap=_OverflowingFloat())


def test_apply_absence_adjustment_rejects_overflowing_session_gap() -> None:
    with pytest.raises(ValueError, match="session_gap"):
        apply_absence_adjustment(
            np.zeros((1, 1), dtype=float),
            _plane(),
            _plane(),
            session_gap=_OverflowingFloat(),
        )
