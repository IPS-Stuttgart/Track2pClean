from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.absence_model import (
    absence_cost_vector,
    apply_absence_adjustment,
    gap_penalty_matrix,
)


class _OverflowingIndex:
    def __index__(self) -> int:
        raise OverflowError("index adapter overflow")


class _BadIndex:
    def __index__(self) -> int:
        raise ValueError("adapter-specific validation failure")


def _plane(n_rois: object = 1) -> SimpleNamespace:
    return SimpleNamespace(
        n_rois=n_rois,
        cell_probabilities=None,
        traces=np.zeros((1, 1), dtype=float),
        spike_traces=None,
    )


@pytest.mark.parametrize("n_rois", [_OverflowingIndex(), _BadIndex()])
def test_absence_cost_vector_normalizes_bad_roi_count_index_adapters(
    n_rois: object,
) -> None:
    with pytest.raises(ValueError, match=r"plane\.n_rois"):
        absence_cost_vector(_plane(n_rois))


def test_gap_penalty_matrix_normalizes_reference_roi_count_overflow() -> None:
    with pytest.raises(ValueError, match=r"reference_plane\.n_rois"):
        gap_penalty_matrix(_plane(_OverflowingIndex()), _plane())


def test_apply_absence_adjustment_normalizes_measurement_roi_count_overflow() -> None:
    with pytest.raises(ValueError, match=r"measurement_plane\.n_rois"):
        apply_absence_adjustment(
            np.zeros((1, 1), dtype=float),
            _plane(),
            _plane(_OverflowingIndex()),
        )
