from __future__ import annotations

import pytest

from bayescatrack.experiments import benchmark_manifest as bm


def test_manifest_float_option_rejects_boolean_values():
    with pytest.raises(ValueError, match="iou_distance_threshold"):
        bm._float_option(  # pylint: disable=protected-access
            {"iou_distance_threshold": True},
            "iou_distance_threshold",
            default=12.0,
        )


def test_manifest_float_option_accepts_numeric_strings():
    value = bm._float_option(  # pylint: disable=protected-access
        {"iou_distance_threshold": "7.5"},
        "iou_distance_threshold",
        default=12.0,
    )

    assert value == 7.5
