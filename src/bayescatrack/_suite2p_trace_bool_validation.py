"""Strict public trace-loader boolean validation for Suite2p loading."""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

_TRACE_BOOL_DEFAULTS: dict[str, bool] = {
    "load_traces": True,
    "load_spike_traces": True,
    "load_neuropil_traces": False,
}
_PLANE_PATCH_ATTR = "_bayescatrack_suite2p_trace_bool_validation_patch"
_SUBJECT_PATCH_ATTR = "_bayescatrack_subject_suite2p_trace_bool_validation_patch"
_EXPORT_PATCH_ATTR = "_bayescatrack_export_suite2p_trace_bool_validation_patch"
_SUMMARY_PATCH_ATTR = "_bayescatrack_summary_suite2p_trace_bool_validation_patch"


def install_suite2p_trace_bool_validation(bridge_module: Any) -> None:
    """Keep trace-loading controls strict after stat-validation compatibility patches."""

    original_load_suite2p_plane = bridge_module.load_suite2p_plane
    if not getattr(original_load_suite2p_plane, _PLANE_PATCH_ATTR, False):

        @wraps(original_load_suite2p_plane)
        def load_suite2p_plane(plane_dir: str | Path, *args: Any, **kwargs: Any) -> Any:
            if not args:
                _validate_trace_bool_kwargs(kwargs)
            return original_load_suite2p_plane(plane_dir, *args, **kwargs)

        setattr(load_suite2p_plane, _PLANE_PATCH_ATTR, True)
        setattr(
            load_suite2p_plane, "_bayescatrack_original", original_load_suite2p_plane
        )
        bridge_module.load_suite2p_plane = load_suite2p_plane

    _install_subject_like_validation(
        bridge_module,
        "load_track2p_subject",
        _SUBJECT_PATCH_ATTR,
        max_extra_positional=0,
    )
    _install_subject_like_validation(
        bridge_module,
        "export_subject_to_npz",
        _EXPORT_PATCH_ATTR,
        max_extra_positional=1,
    )
    _install_subject_like_validation(
        bridge_module,
        "summarize_subject",
        _SUMMARY_PATCH_ATTR,
        max_extra_positional=0,
    )


def _install_subject_like_validation(
    bridge_module: Any,
    name: str,
    patch_attr: str,
    *,
    max_extra_positional: int,
) -> None:
    original = getattr(bridge_module, name)
    if getattr(original, patch_attr, False):
        return

    @wraps(original)
    def subject_like_entrypoint(
        subject_dir: str | Path, *args: Any, **kwargs: Any
    ) -> Any:
        if len(args) <= max_extra_positional and _uses_suite2p_input_format(kwargs):
            _validate_trace_bool_kwargs(kwargs)
        return original(subject_dir, *args, **kwargs)

    setattr(subject_like_entrypoint, patch_attr, True)
    setattr(subject_like_entrypoint, "_bayescatrack_original", original)
    setattr(bridge_module, name, subject_like_entrypoint)


def _uses_suite2p_input_format(kwargs: dict[str, Any]) -> bool:
    return kwargs.get("input_format", "auto") in {"auto", "suite2p"}


def _validate_trace_bool_kwargs(kwargs: dict[str, Any]) -> None:
    for name, default in _TRACE_BOOL_DEFAULTS.items():
        value = kwargs.get(name, default)
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")


__all__ = ["install_suite2p_trace_bool_validation"]
