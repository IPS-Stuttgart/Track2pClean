"""Strict validation hooks for Track2p reference helpers."""

from __future__ import annotations

from types import ModuleType
from typing import Any, Iterable

import numpy as np

_PATCH_ATTR = "_bayescatrack_reference_validation_patch"


def install_reference_validation(reference_module: ModuleType | None = None) -> None:
    """Install idempotent validation wrappers on ``bayescatrack.reference``."""

    if reference_module is None:
        from . import (
            reference as reference_module,  # pylint: disable=import-outside-toplevel,reimported
        )

    _install_optional_int_parser_validation(reference_module)
    _install_curated_mask_validation(reference_module)
    _install_curated_only_validation(reference_module)
    _install_session_index_validation(reference_module)


def _install_optional_int_parser_validation(reference_module: ModuleType) -> None:
    original_parse_optional_int = reference_module._parse_optional_int  # pylint: disable=protected-access
    if getattr(original_parse_optional_int, _PATCH_ATTR, False):
        return

    missing_strings = frozenset(reference_module._MISSING_STRINGS)  # pylint: disable=protected-access

    def _parse_optional_int_with_validation(value: Any) -> int | None:
        parsed_value = original_parse_optional_int(value)
        if parsed_value is not None or _is_explicit_missing_roi_index(
            value,
            missing_strings=missing_strings,
        ):
            return parsed_value
        raise ValueError(
            "ROI index must be integer-like or an explicit missing value; "
            f"got {value!r}"
        )

    setattr(_parse_optional_int_with_validation, _PATCH_ATTR, True)
    setattr(
        _parse_optional_int_with_validation,
        "_bayescatrack_original",
        original_parse_optional_int,
    )
    reference_module._parse_optional_int = _parse_optional_int_with_validation  # pylint: disable=protected-access


def _install_curated_mask_validation(reference_module: ModuleType) -> None:
    reference_cls = reference_module.Track2pReference
    original_post_init = reference_cls.__post_init__
    if getattr(original_post_init, _PATCH_ATTR, False):
        return

    def _post_init_with_reference_validation(self: Any) -> None:
        if self.curated_mask is not None:
            indices = reference_module._as_nullable_int_matrix(  # pylint: disable=protected-access
                self.suite2p_indices
            )
            object.__setattr__(
                self,
                "curated_mask",
                _normalize_curated_mask(
                    self.curated_mask,
                    n_tracks=int(indices.shape[0]),
                ),
            )
        original_post_init(self)

    setattr(_post_init_with_reference_validation, _PATCH_ATTR, True)
    setattr(
        _post_init_with_reference_validation,
        "_bayescatrack_original",
        original_post_init,
    )
    reference_cls.__post_init__ = _post_init_with_reference_validation


def _install_curated_only_validation(reference_module: ModuleType) -> None:
    reference_cls = reference_module.Track2pReference
    original_filtered_indices = reference_cls._filtered_indices
    if getattr(original_filtered_indices, _PATCH_ATTR, False):
        return

    def _filtered_indices_with_curated_only_validation(
        self: Any,
        *,
        curated_only: Any = False,
    ) -> np.ndarray:
        return original_filtered_indices(
            self,
            curated_only=_coerce_curated_only_flag(curated_only),
        )

    setattr(_filtered_indices_with_curated_only_validation, _PATCH_ATTR, True)
    setattr(
        _filtered_indices_with_curated_only_validation,
        "_bayescatrack_original",
        original_filtered_indices,
    )
    reference_cls._filtered_indices = _filtered_indices_with_curated_only_validation


def _install_session_index_validation(reference_module: ModuleType) -> None:
    original_normalize_session_indices = (
        reference_module._normalize_session_indices
    )  # pylint: disable=protected-access
    if not getattr(original_normalize_session_indices, _PATCH_ATTR, False):

        def _normalize_session_indices_with_validation(
            session_indices: Iterable[Any] | None,
            n_sessions: int,
        ) -> tuple[int, ...]:
            if session_indices is None:
                return original_normalize_session_indices(session_indices, n_sessions)
            if isinstance(session_indices, (str, bytes)):
                raise ValueError(
                    "session_indices must be an iterable of integer session indices"
                )
            normalized = tuple(
                _coerce_session_index(
                    session_index,
                    context="session_indices",
                    allow_integer_like=True,
                )
                for session_index in session_indices
            )
            return original_normalize_session_indices(normalized, n_sessions)

        setattr(_normalize_session_indices_with_validation, _PATCH_ATTR, True)
        setattr(
            _normalize_session_indices_with_validation,
            "_bayescatrack_original",
            original_normalize_session_indices,
        )
        reference_module._normalize_session_indices = _normalize_session_indices_with_validation  # pylint: disable=protected-access

    original_validate_session_index = (
        reference_module._validate_session_index
    )  # pylint: disable=protected-access
    if getattr(original_validate_session_index, _PATCH_ATTR, False):
        return

    def _validate_session_index_with_validation(
        session_index: Any, n_sessions: int
    ) -> None:
        normalized = _coerce_session_index(
            session_index,
            context="session index",
            allow_integer_like=False,
        )
        original_validate_session_index(normalized, n_sessions)

    setattr(_validate_session_index_with_validation, _PATCH_ATTR, True)
    setattr(
        _validate_session_index_with_validation,
        "_bayescatrack_original",
        original_validate_session_index,
    )
    reference_module._validate_session_index = (
        _validate_session_index_with_validation  # pylint: disable=protected-access
    )


def _is_explicit_missing_roi_index(
    value: Any,
    *,
    missing_strings: frozenset[str],
) -> bool:
    if value is None:
        return True
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in missing_strings:
            return True
        try:
            numeric_value = float(stripped)
        except ValueError:
            return False
        return _is_numeric_missing_roi_index(numeric_value)
    if isinstance(value, (int, np.integer)):
        return int(value) < 0
    if isinstance(value, (float, np.floating)):
        return _is_numeric_missing_roi_index(float(value))
    return False


def _is_numeric_missing_roi_index(numeric_value: float) -> bool:
    if np.isnan(numeric_value):
        return True
    return bool(
        np.isfinite(numeric_value)
        and numeric_value.is_integer()
        and numeric_value < 0.0
    )


def _normalize_curated_mask(mask: Any, *, n_tracks: int) -> np.ndarray:
    array = np.asarray(mask, dtype=object).reshape(-1)
    if array.shape != (n_tracks,):
        raise ValueError("curated_mask must have shape (n_tracks,)")

    normalized = np.empty(array.shape, dtype=bool)
    for index, value in enumerate(array.tolist()):
        normalized[index] = _coerce_curated_mask_value(value, location=index)
    return normalized


def _coerce_curated_mask_value(value: Any, *, location: int) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        if integer in {0, 1}:
            return bool(integer)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(numeric)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(
        "curated_mask contains a non-boolean value "
        f"at index {location}: {value!r}; expected booleans or explicit 0/1 values"
    )


def _coerce_curated_only_flag(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError("curated_only must be a boolean")


def _coerce_session_index(
    value: Any,
    *,
    context: str,
    allow_integer_like: bool,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{context} must contain integer session indices, got boolean {value!r}"
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    if allow_integer_like and isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    if allow_integer_like and isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(
                f"{context} must contain integer session indices, got {value!r}"
            ) from exc
    raise ValueError(f"{context} must contain integer session indices, got {value!r}")
