"""Strict validation for Track2p reference session names."""

from __future__ import annotations

from functools import wraps
from typing import Any

_PATCH_MARKER = "_bayescatrack_reference_session_name_validation_patch"


def install_reference_session_name_validation() -> None:
    """Install an idempotent uniqueness guard for Track2pReference sessions."""

    from . import reference as reference_module  # pylint: disable=import-outside-toplevel

    reference_cls = reference_module.Track2pReference
    original_post_init = reference_cls.__post_init__
    if getattr(original_post_init, _PATCH_MARKER, False):
        return

    @wraps(original_post_init)
    def _post_init_with_unique_session_names(self: Any) -> None:
        session_names = tuple(str(name) for name in self.session_names)
        if len(set(session_names)) != len(session_names):
            raise ValueError("session_names must be unique")
        original_post_init(self)

    setattr(_post_init_with_unique_session_names, _PATCH_MARKER, True)
    setattr(
        _post_init_with_unique_session_names,
        "_bayescatrack_original",
        original_post_init,
    )
    reference_cls.__post_init__ = _post_init_with_unique_session_names


__all__ = ["install_reference_session_name_validation"]
