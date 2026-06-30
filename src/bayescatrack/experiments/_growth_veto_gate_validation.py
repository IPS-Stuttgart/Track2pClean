"""Strict validation for growth-veto gate structural flags."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Any

_PATCH_MARKER = "_bayescatrack_growth_veto_gate_validation_patch"
_STRUCTURAL_FLAG_NAMES = (
    "require_not_suffix_edge",
    "require_terminal_edge",
    "require_last_session_edge",
    "require_complete_component",
)


def install_growth_veto_gate_validation() -> None:
    """Install idempotent boolean validation for ``GrowthVetoGate`` flags.

    The growth-veto cleanup runner uses these fields as structural switches when
    deciding whether an accepted edge is eligible for splitting.  Programmatic
    configs must therefore pass real booleans; values such as ``"false"`` or
    ``1`` are otherwise truthy and silently enable a gate the caller may have
    intended to disable.
    """

    cleanup = import_module("bayescatrack.experiments.track2p_policy_growth_veto_cleanup")
    original_post_init = cleanup.GrowthVetoGate.__post_init__
    if getattr(original_post_init, _PATCH_MARKER, False):
        return

    @wraps(original_post_init)
    def __post_init_with_boolean_validation(self: Any) -> None:
        original_post_init(self)
        for field_name in _STRUCTURAL_FLAG_NAMES:
            object.__setattr__(
                self,
                field_name,
                _strict_bool_value(getattr(self, field_name), name=field_name),
            )

    setattr(__post_init_with_boolean_validation, _PATCH_MARKER, True)
    setattr(__post_init_with_boolean_validation, "_bayescatrack_original", original_post_init)
    cleanup.GrowthVetoGate.__post_init__ = __post_init_with_boolean_validation


def _strict_bool_value(value: Any, *, name: str) -> bool:
    if type(value) is bool:
        return value
    raise ValueError(f"{name} must be a boolean")


__all__ = ["install_growth_veto_gate_validation"]
