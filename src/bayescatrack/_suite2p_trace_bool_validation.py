"""Strict public trace-loader boolean validation for Suite2p loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_TRACE_BOOL_DEFAULTS: dict[str, bool] = {
    "load_traces": True,
    "load_spike_traces": True,
    "load_neuropil_traces": False,
}
_PATCH_ATTR = "_bayescatrack_suite2p_trace_bool_validation_patch"


def install_suite2p_trace_bool_validation(bridge_module: Any) -> None:
    """Keep trace-loading controls strict after stat-validation compatibility patches."""

    original_load_suite2p_plane = bridge_module.load_suite2p_plane
    if getattr(original_load_suite2p_plane, _PATCH_ATTR, False):
        return

    def load_suite2p_plane(plane_dir: str | Path, *args: Any, **kwargs: Any) -> Any:
        if not args:
            for name, default in _TRACE_BOOL_DEFAULTS.items():
                value = kwargs.get(name, default)
                if type(value) is not bool:
                    raise ValueError(f"{name} must be a boolean")
        return original_load_suite2p_plane(plane_dir, *args, **kwargs)

    setattr(load_suite2p_plane, _PATCH_ATTR, True)
    setattr(load_suite2p_plane, "_bayescatrack_original", original_load_suite2p_plane)
    bridge_module.load_suite2p_plane = load_suite2p_plane


__all__ = ["install_suite2p_trace_bool_validation"]
