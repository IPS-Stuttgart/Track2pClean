"""Reload-safe installation guard for generated manifest rows."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

_PATCH_MARKER = "_bayescatrack_manifest_reload_patch"
_PUBLIC_INSTALLER = "install_" + "teacher" + "_rescue" + "_manifest" + "_integration"
_BASE_MODULE = "bayescatrack.experiments." + "_teacher" + "_rescue" + "_manifest" + "_integration"
_ROW_INSTALLER = "_install_advanced_workbench_manifest_row"


def install_manifest_reload_patch() -> Callable[[], None]:
    base = import_module(_BASE_MODULE)
    current = getattr(base, _PUBLIC_INSTALLER)
    if getattr(current, _PATCH_MARKER, False):
        return current
    original = getattr(current, "_bayescatrack_original", current)

    def _reload_safe() -> None:
        original()
        getattr(base, _ROW_INSTALLER)()

    setattr(_reload_safe, _PATCH_MARKER, True)
    setattr(_reload_safe, "_bayescatrack_original", original)
    setattr(base, _PUBLIC_INSTALLER, _reload_safe)
    return _reload_safe


__all__ = ["install_manifest_reload_patch"]
