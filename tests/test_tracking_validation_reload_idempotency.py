from __future__ import annotations

import importlib
from typing import Any

import bayescatrack
import bayescatrack.tracking as tracking


def _wrapper_count(function: Any, marker: str) -> int:
    count = 0
    seen: set[int] = set()
    current = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            raise AssertionError("cycle in tracking validation wrapper chain")
        seen.add(current_id)
        if getattr(current, marker, False):
            count += 1
        current = getattr(current, "_bayescatrack_original", None)
    return count


def test_tracking_validation_patch_installers_are_reload_idempotent():
    importlib.reload(bayescatrack)
    importlib.reload(bayescatrack)

    assert _wrapper_count(
        tracking.run_registered_subject_tracking,
        "_bayescatrack_start_roi_validation_patch",
    ) == 1
    assert _wrapper_count(
        tracking.run_registered_subject_tracking,
        "_bayescatrack_tracking_start_roi_availability_validation_patch",
    ) == 1
    assert _wrapper_count(
        tracking._restrict_track_rows_to_start_rois,
        "_bayescatrack_start_roi_validation_patch",
    ) == 1
    assert _wrapper_count(
        tracking._restrict_track_rows_to_start_rois,
        "_bayescatrack_tracking_duplicate_start_roi_validation_patch",
    ) == 1
