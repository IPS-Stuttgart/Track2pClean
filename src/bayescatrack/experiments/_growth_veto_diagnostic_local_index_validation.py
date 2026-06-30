"""Guard growth-veto diagnostic helpers against below-zero local ROI indices.

Track2p diagnostics store ROI positions local to adjacent Suite2p ROI-index
vectors.  The growth-veto helpers already ignore local indices that are too
large, but Python sequence indexing maps below-zero values to entries from the
end of the ROI vector.  A malformed diagnostic could therefore be interpreted as
a real edge feature or growth anchor.  This installer filters those malformed
entries before they reach the helper implementations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import wraps
from typing import Any

_PATCH_MARKER = "_bayescatrack_growth_veto_local_index_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"


def install_growth_veto_diagnostic_local_index_validation() -> None:
    """Install idempotent local-index validation for growth-veto diagnostics."""

    from . import track2p_policy_growth_veto_whatif as _veto  # pylint: disable=import-outside-toplevel

    _install_anchor_edge_wrapper(_veto)
    _install_policy_feature_wrapper(_veto)


def _install_anchor_edge_wrapper(_veto: Any) -> None:
    original = _veto._anchor_edges_from_policy_diagnostics
    if _has_patch(original):
        return

    @wraps(original)
    def _anchor_edges_with_local_index_guard(
        sessions: Sequence[Any],
        *args: Any,
        diagnostics: Iterable[Any],
        **kwargs: Any,
    ) -> Any:
        return original(
            sessions,
            *args,
            diagnostics=_valid_local_index_diagnostics(diagnostics),
            **kwargs,
        )

    setattr(_anchor_edges_with_local_index_guard, _PATCH_MARKER, True)
    setattr(_anchor_edges_with_local_index_guard, _ORIGINAL_ATTR, original)
    _veto._anchor_edges_from_policy_diagnostics = _anchor_edges_with_local_index_guard


def _install_policy_feature_wrapper(_veto: Any) -> None:
    original = _veto._policy_feature_index_from_diagnostics
    if _has_patch(original):
        return

    @wraps(original)
    def _policy_feature_index_with_local_index_guard(
        sessions: Sequence[Any],
        diagnostics: Iterable[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return original(
            sessions,
            _valid_local_index_diagnostics(diagnostics),
            *args,
            **kwargs,
        )

    setattr(_policy_feature_index_with_local_index_guard, _PATCH_MARKER, True)
    setattr(_policy_feature_index_with_local_index_guard, _ORIGINAL_ATTR, original)
    _veto._policy_feature_index_from_diagnostics = _policy_feature_index_with_local_index_guard


def _valid_local_index_diagnostics(diagnostics: Iterable[Any]) -> tuple[Any, ...]:
    """Return diagnostics whose local ROI positions cannot wrap around."""

    output: list[Any] = []
    for diagnostic in diagnostics:
        if int(diagnostic.local_roi_a) < 0 or int(diagnostic.local_roi_b) < 0:
            continue
        output.append(diagnostic)
    return tuple(output)


def _has_patch(function: Any) -> bool:
    current = function
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if getattr(current, _PATCH_MARKER, False):
            return True
        visited.add(id(current))
        current = getattr(current, _ORIGINAL_ATTR, None)
    return False


__all__ = ["install_growth_veto_diagnostic_local_index_validation"]
