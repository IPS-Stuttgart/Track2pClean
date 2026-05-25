from __future__ import annotations

import numpy as np

from bayescatrack.experiments.track2p_emulation_benchmark import (
    _threshold_assigned_iou,
)


def test_degenerate_positive_iou_threshold_keeps_identical_matches() -> None:
    assigned_iou = np.asarray([1.0, 1.0], dtype=float)

    threshold = _threshold_assigned_iou(assigned_iou, method="otsu")

    assert threshold < 1.0
    assert np.all(assigned_iou > threshold)


def test_degenerate_zero_iou_threshold_still_rejects_zero_matches() -> None:
    assigned_iou = np.asarray([0.0, 0.0], dtype=float)

    threshold = _threshold_assigned_iou(assigned_iou, method="otsu")

    assert threshold == 0.0
    assert not np.any(assigned_iou > threshold)
