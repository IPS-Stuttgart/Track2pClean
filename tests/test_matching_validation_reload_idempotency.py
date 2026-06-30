from __future__ import annotations

import importlib
from typing import Any

import bayescatrack
from bayescatrack import matching


def _patch_count(function: Any, marker: str) -> int:
    count = 0
    seen: set[int] = set()
    current: Any = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            raise AssertionError("cycle in matching wrapper chain")
        seen.add(current_id)
        if getattr(current, marker, False):
            count += 1
        current = getattr(current, "_bayescatrack_original", None)
    return count


def test_matching_solver_validation_wrappers_are_reload_idempotent():
    importlib.reload(bayescatrack)
    importlib.reload(bayescatrack)

    solve = matching.solve_bundle_linear_assignment
    assert _patch_count(solve, "_bayescatrack_assignment_bundle_validation_patch") == 1
    assert _patch_count(solve, "_bayescatrack_bundle_roi_index_validation_patch") == 1
    assert _patch_count(solve, "_bayescatrack_matching_control_validation_patch") == 1

    control_marker = "_bayescatrack_matching_control_validation_patch"
    assert _patch_count(matching.build_track_rows_from_matches, control_marker) == 1
    assert _patch_count(matching._bundle_roi_indices_for_session, control_marker) == 1
    assert _patch_count(matching._normalize_roi_index, control_marker) == 1
    assert _patch_count(matching._normalize_session_index, control_marker) == 1
