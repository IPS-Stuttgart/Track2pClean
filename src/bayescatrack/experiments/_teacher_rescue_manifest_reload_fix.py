"""Reload-safe installation guard for teacher-rescue manifest rows."""

from __future__ import annotations

from collections.abc import Callable

_PATCH_MARKER = "_bayescatrack_teacher_rescue_manifest_reload_fix"


def install_teacher_rescue_manifest_reload_fix() -> Callable[[], None]:
    """Patch the public teacher-rescue installer to survive workbench reloads."""

    from bayescatrack.experiments import _teacher_rescue_manifest_integration as base

    current = base.install_teacher_rescue_manifest_integration
    if getattr(current, _PATCH_MARKER, False):
        return current
    original = getattr(current, "_bayescatrack_original", current)

    def _install_teacher_rescue_manifest_reload_safe() -> None:
        from bayescatrack.experiments import benchmark_manifest as manifest

        if getattr(manifest, "_bayescatrack_teacher_rescue_manifest_integration", False):
            base._install_advanced_workbench_manifest_row()
            return
        original()

    setattr(_install_teacher_rescue_manifest_reload_safe, _PATCH_MARKER, True)
    setattr(_install_teacher_rescue_manifest_reload_safe, "_bayescatrack_original", original)
    base.install_teacher_rescue_manifest_integration = (
        _install_teacher_rescue_manifest_reload_safe
    )
    return _install_teacher_rescue_manifest_reload_safe


__all__ = ["install_teacher_rescue_manifest_reload_fix"]
