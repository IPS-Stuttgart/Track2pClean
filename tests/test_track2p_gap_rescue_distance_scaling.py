from __future__ import annotations

import numpy as np

from bayescatrack.experiments import track2p_emulation_benchmark as bench


def test_gap_rescue_scales_distance_threshold_for_direct_skips(monkeypatch) -> None:
    calls: list[float] = []

    def fake_links(reference_session, moving_session, **kwargs):
        del reference_session, moving_session
        calls.append(float(kwargs["iou_distance_threshold"]))
        return np.zeros((0, 2), dtype=int)

    monkeypatch.setattr(bench, "_thresholded_hungarian_links", fake_links)

    bench._thresholded_links_by_gap(
        ("s0", "s1", "s2"),
        transform_type="affine",
        threshold_method="min",
        iou_distance_threshold=12.0,
        max_gap=2,
    )

    assert calls == [12.0, 24.0, 12.0]
