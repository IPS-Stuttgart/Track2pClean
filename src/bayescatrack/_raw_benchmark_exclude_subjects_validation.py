"""Normalize raw benchmark excluded-subject inputs."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import Any

_PATCH_MARKER = "_bayescatrack_raw_benchmark_exclude_subjects_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_raw_benchmark_exclude_subjects_original"


def install_raw_benchmark_exclude_subjects_validation() -> None:
    """Install idempotent normalization for raw-benchmark subject excludes."""

    from bayescatrack.experiments import track2p_raw_benchmark_data as raw_benchmark  # pylint: disable=import-outside-toplevel

    current_prepare = raw_benchmark.prepare_raw_suite2p_benchmark_data
    if getattr(current_prepare, _PATCH_MARKER, False):
        return

    original_prepare = getattr(current_prepare, _ORIGINAL_ATTR, current_prepare)

    @wraps(original_prepare)
    def prepare_raw_suite2p_benchmark_data(*args: Any, **kwargs: Any) -> Any:
        normalized_kwargs = dict(kwargs)
        normalized_kwargs["exclude_subjects"] = _normalize_exclude_subjects(
            normalized_kwargs.get("exclude_subjects", ())
        )
        return original_prepare(*args, **normalized_kwargs)

    setattr(prepare_raw_suite2p_benchmark_data, _PATCH_MARKER, True)
    setattr(prepare_raw_suite2p_benchmark_data, _ORIGINAL_ATTR, original_prepare)
    raw_benchmark.prepare_raw_suite2p_benchmark_data = prepare_raw_suite2p_benchmark_data


def _normalize_exclude_subjects(exclude_subjects: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(exclude_subjects, str):
        stripped = exclude_subjects.strip()
        return () if not stripped else (stripped,)
    if isinstance(exclude_subjects, (bytes, bytearray)):
        raise ValueError("exclude_subjects must be a string or an iterable of strings")

    try:
        iterator = iter(exclude_subjects)
    except TypeError as exc:
        raise ValueError("exclude_subjects must be a string or an iterable of strings") from exc

    normalized: list[str] = []
    for subject_name in iterator:
        if not isinstance(subject_name, str):
            raise ValueError("exclude_subjects entries must be strings")
        stripped = subject_name.strip()
        if stripped:
            normalized.append(stripped)
    return tuple(normalized)


__all__ = ["install_raw_benchmark_exclude_subjects_validation"]
