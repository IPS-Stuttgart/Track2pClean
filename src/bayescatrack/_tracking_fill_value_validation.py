"""Strict validation for tracking-result missing-value sentinels.

Tracking-level helpers use ``fill_value`` to distinguish missing cells from
Suite2p ROI identifiers.  Suite2p ROI identifiers are non-negative, so malformed
or non-negative sentinels can silently turn a real ROI into a missing detection
when computing track lengths, restricting global tracks, or exporting link-cost
matrices.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_tracking_fill_value_validation_patch"
_ERROR_MESSAGE = "fill_value must be a negative integer sentinel"


def install_tracking_fill_value_validation() -> None:
    """Install idempotent validation around tracking missing sentinels."""

    from . import tracking as _tracking  # pylint: disable=import-outside-toplevel

    original_post_init = _tracking.SubjectTrackingResult.__post_init__
    if not getattr(original_post_init, _PATCH_MARKER, False):

        @wraps(original_post_init)
        def subject_tracking_result_post_init_with_fill_value_validation(
            self: Any,
        ) -> Any:
            object.__setattr__(
                self,
                "fill_value",
                _normalize_fill_value(getattr(self, "fill_value", -1)),
            )
            return original_post_init(self)

        setattr(
            subject_tracking_result_post_init_with_fill_value_validation,
            _PATCH_MARKER,
            True,
        )
        setattr(
            subject_tracking_result_post_init_with_fill_value_validation,
            "_bayescatrack_original",
            original_post_init,
        )
        _tracking.SubjectTrackingResult.__post_init__ = (
            subject_tracking_result_post_init_with_fill_value_validation
        )

    _patch_fill_value_keyword_function(_tracking, "run_registered_subject_tracking")
    _patch_fill_value_keyword_function(_tracking, "_coerce_global_track_rows")
    _patch_fill_value_keyword_function(_tracking, "_restrict_track_rows_to_start_rois")
    _patch_fill_value_keyword_function(_tracking, "_build_link_cost_matrix")
    _patch_fill_value_keyword_function(_tracking, "_build_global_link_cost_matrices")
    _patch_fill_value_keyword_function(_tracking, "_default_link_target_indices")


def _patch_fill_value_keyword_function(module: Any, name: str) -> None:
    original = getattr(module, name)
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def function_with_fill_value_validation(*args: Any, **kwargs: Any) -> Any:
        if "fill_value" in kwargs:
            kwargs = dict(kwargs)
            kwargs["fill_value"] = _normalize_fill_value(kwargs["fill_value"])
        return original(*args, **kwargs)

    setattr(function_with_fill_value_validation, _PATCH_MARKER, True)
    setattr(function_with_fill_value_validation, "_bayescatrack_original", original)
    setattr(module, name, function_with_fill_value_validation)


def _normalize_fill_value(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_ERROR_MESSAGE)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_ERROR_MESSAGE)
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(_ERROR_MESSAGE) from exc

    integer_value = int(integer_value)
    if integer_value >= 0:
        raise ValueError(
            "fill_value must be a negative integer sentinel that cannot collide "
            "with non-negative ROI indices"
        )
    return integer_value


__all__ = ["install_tracking_fill_value_validation"]
