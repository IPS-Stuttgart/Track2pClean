"""Validation for prior config normalizers.

Several optional config normalizers are used by manifest/CLI entry points where
``dict`` inputs are common.  Calling ``dict(value)`` before verifying that
``value`` is a mapping makes malformed controls ambiguous: empty tuples/lists are
silently accepted as default configs, while other non-mapping values leak low-level
``dict`` construction errors.  Normalize these helpers to reject non-mapping
config inputs with a stable user-facing ``ValueError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

_PATCH_MARKER = "_bayescatrack_prior_config_mapping_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_prior_config_mapping_validation_original"


def install_prior_config_mapping_validation() -> None:
    """Install idempotent mapping validation on remaining prior config helpers."""

    from . import (
        absence_model as _absence_model,  # pylint: disable=import-outside-toplevel
    )
    from . import (
        teacher_priors as _teacher_priors,  # pylint: disable=import-outside-toplevel
    )
    from . import (
        track2p_policy_priors as _track2p_policy_priors,  # pylint: disable=import-outside-toplevel
    )

    _patch_config_normalizer(
        _absence_model,
        function_name="absence_model_config_from_mapping",
        config_cls=_absence_model.AbsenceModelConfig,
        config_name="AbsenceModelConfig",
    )
    _patch_config_normalizer(
        _teacher_priors,
        function_name="teacher_edge_prior_config_from_mapping",
        config_cls=_teacher_priors.TeacherEdgePriorConfig,
        config_name="TeacherEdgePriorConfig",
    )
    _patch_config_normalizer(
        _track2p_policy_priors,
        function_name="track2p_policy_prior_config_from_mapping",
        config_cls=_track2p_policy_priors.Track2pPolicyPriorConfig,
        config_name="Track2pPolicyPriorConfig",
    )


def _patch_config_normalizer(
    module: Any,
    *,
    function_name: str,
    config_cls: type[Any],
    config_name: str,
) -> None:
    original = getattr(module, function_name)
    if _method_chain_has_patch(original):
        return

    @wraps(original)
    def config_from_mapping_with_mapping_validation(value: Any) -> Any:
        if value is None or isinstance(value, config_cls):
            return original(value)
        if not isinstance(value, Mapping):
            raise ValueError(
                f"config must be None, a {config_name}, or a mapping of "
                f"{config_name} fields"
            )
        return original(value)

    setattr(config_from_mapping_with_mapping_validation, _PATCH_MARKER, True)
    setattr(config_from_mapping_with_mapping_validation, _ORIGINAL_ATTR, original)
    setattr(module, function_name, config_from_mapping_with_mapping_validation)


def _method_chain_has_patch(method: Any) -> bool:
    seen: set[int] = set()
    current: Any = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, _PATCH_MARKER, False):
            return True
        seen.add(current_id)
        current = getattr(current, _ORIGINAL_ATTR, None)
    return False


__all__ = ["install_prior_config_mapping_validation"]
