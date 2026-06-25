"""Strict public trace-loader boolean validation for Suite2p loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_TRACE_BOOL_DEFAULTS: dict[str, bool] = {
    "load_traces": True,
    "load_spike_traces": True,
    "load_neuropil_traces": False,
}
_PLANE_PATCH_ATTR = "_bayescatrack_suite2p_trace_bool_validation_patch"
_SUBJECT_PATCH_ATTR = "_bayescatrack_subject_suite2p_trace_bool_validation_patch"


def install_suite2p_trace_bool_validation(bridge_module: Any) -> None:
    """Keep trace-loading controls strict after stat-validation compatibility patches."""

    original_load_suite2p_plane = bridge_module.load_suite2p_plane
    if not getattr(original_load_suite2p_plane, _PLANE_PATCH_ATTR, False):

        def load_suite2p_plane(plane_dir: str | Path, *args: Any, **kwargs: Any) -> Any:
            if not args:
                _validate_trace_bool_kwargs(kwargs)
            return original_load_suite2p_plane(plane_dir, *args, **kwargs)

        setattr(load_suite2p_plane, _PLANE_PATCH_ATTR, True)
        setattr(load_suite2p_plane, "_bayescatrack_original", original_load_suite2p_plane)
        bridge_module.load_suite2p_plane = load_suite2p_plane

    original_load_track2p_subject = bridge_module.load_track2p_subject
    if not getattr(original_load_track2p_subject, _SUBJECT_PATCH_ATTR, False):

        def load_track2p_subject(subject_dir: str | Path, *args: Any, **kwargs: Any) -> Any:
            if not args and kwargs.get("input_format", "auto") in {"auto", "suite2p"}:
                _validate_trace_bool_kwargs(kwargs)
            return original_load_track2p_subject(subject_dir, *args, **kwargs)

        setattr(load_track2p_subject, _SUBJECT_PATCH_ATTR, True)
        setattr(load_track2p_subject, "_bayescatrack_original", original_load_track2p_subject)
        bridge_module.load_track2p_subject = load_track2p_subject


def _validate_trace_bool_kwargs(kwargs: dict[str, Any]) -> None:
    for name, default in _TRACE_BOOL_DEFAULTS.items():
        value = kwargs.get(name, default)
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")


__all__ = ["install_suite2p_trace_bool_validation"]
