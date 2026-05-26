"""Regression tests for Track2p-policy threshold robustness."""

from __future__ import annotations

import numpy as np
from bayescatrack.experiments.track2p_emulation_benchmark import (
    _threshold_assigned_iou,
)


def test_minimum_threshold_falls_back_for_monotone_positive_iou_values() -> None:
    assigned_iou = np.asarray([0.10, 0.11, 0.12, 0.13], dtype=float)

    threshold = _threshold_assigned_iou(assigned_iou, method="min")

    assert np.isfinite(threshold)
    assert float(assigned_iou.min()) <= threshold <= float(assigned_iou.max())
