"""Normalize malformed ``__index__`` failures in FOV-affine validation.

The main FOV-affine validation hook catches ordinary ``TypeError`` failures from
``operator.index``.  Custom index-like controls can instead raise ``ValueError``
or arithmetic errors, which should still surface as the public FOV-affine control
validation messages rather than leaking implementation-specific exceptions.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

_INDEX_ERROR_PATCH_MARKER = "_bayescatrack_fov_affine_index_error_patch"
_INDEX_PROTOCOL_EXCEPTIONS = (ValueError, ArithmeticError)


def install_fov_affine_index_error_validation() -> None:
    """Install idempotent normalization for FOV-affine index-protocol errors."""

    from . import _fov_affine_validation as validation  # pylint: disable=import-outside-toplevel

    original_output_shape_component = validation._normalize_output_shape_component
    if not getattr(original_output_shape_component, _INDEX_ERROR_PATCH_MARKER, False):

        @wraps(original_output_shape_component)
        def _normalize_output_shape_component_with_index_error_normalization(
            value: Any,
        ) -> int:
            try:
                return original_output_shape_component(value)
            except _INDEX_PROTOCOL_EXCEPTIONS as exc:
                raise ValueError(validation._OUTPUT_SHAPE_ERROR) from exc

        _mark_patch(
            _normalize_output_shape_component_with_index_error_normalization,
            original_output_shape_component,
        )
        validation._normalize_output_shape_component = (  # type: ignore[assignment]
            _normalize_output_shape_component_with_index_error_normalization
        )

    original_positive_integer = validation._normalize_positive_integer
    if not getattr(original_positive_integer, _INDEX_ERROR_PATCH_MARKER, False):

        @wraps(original_positive_integer)
        def _normalize_positive_integer_with_index_error_normalization(
            value: Any,
            error_message: str,
        ) -> int:
            try:
                return original_positive_integer(value, error_message)
            except _INDEX_PROTOCOL_EXCEPTIONS as exc:
                raise ValueError(error_message) from exc

        _mark_patch(
            _normalize_positive_integer_with_index_error_normalization,
            original_positive_integer,
        )
        validation._normalize_positive_integer = (  # type: ignore[assignment]
            _normalize_positive_integer_with_index_error_normalization
        )


def _mark_patch(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _INDEX_ERROR_PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


__all__ = ["install_fov_affine_index_error_validation"]
