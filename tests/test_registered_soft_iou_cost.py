"""Regression tests for the registered-soft-iou global-assignment cost."""

from bayescatrack.association.pyrecest_global_assignment import (
    _cost_kwargs_for_method,
    registered_iou_cost_kwargs,
    registered_soft_iou_cost_kwargs,
)


def test_registered_soft_iou_is_dispatcher_supported() -> None:
    assert _cost_kwargs_for_method("registered-soft-iou") == (
        registered_soft_iou_cost_kwargs()
    )


def test_registered_soft_iou_only_switches_overlap_term() -> None:
    registered_iou = registered_iou_cost_kwargs()
    registered_soft_iou = registered_soft_iou_cost_kwargs()

    assert registered_soft_iou["iou_weight"] == 0.0
    assert registered_soft_iou["mask_cosine_weight"] == 1.0

    shared_keys = set(registered_iou) - {"iou_weight", "mask_cosine_weight"}
    for key in shared_keys:
        assert registered_soft_iou[key] == registered_iou[key]
