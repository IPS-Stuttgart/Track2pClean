from __future__ import annotations

import builtins

from bayescatrack._advanced_weight_validation import (
    _REJECTED_SCALAR_TYPES,
    _finite_positive_float,
)


def test_advanced_positive_float_accepts_one() -> None:
    assert _finite_positive_float(1, name="large_cost") == 1.0


def test_advanced_scalar_guard_lists_view_type() -> None:
    view_type = getattr(builtins, "memory" "view")
    assert view_type in _REJECTED_SCALAR_TYPES
