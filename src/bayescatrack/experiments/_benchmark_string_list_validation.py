"""Strict validation for comma-separated benchmark string-list options."""

from __future__ import annotations

from functools import wraps

_PATCH_ATTR = "_track2pclean_benchmark_string_list_validation_patch"


def install_benchmark_string_list_validation() -> None:
    """Install an idempotent validator for benchmark comma-separated lists."""

    from bayescatrack.experiments import track2p_benchmark as benchmark

    original = benchmark._parse_string_list  # pylint: disable=protected-access
    if getattr(original, _PATCH_ATTR, False):
        return

    @wraps(original)
    def _parse_string_list(raw: str | None, *, name: str) -> tuple[str, ...]:
        if raw is None:
            return ()
        if not isinstance(raw, str):
            raise ValueError(f"{name} must be a comma-separated list of strings")
        tokens = tuple(token.strip() for token in raw.split(","))
        if any(not token for token in tokens):
            raise ValueError(
                f"{name} must be a comma-separated list of non-empty values"
            )
        return tokens

    setattr(_parse_string_list, _PATCH_ATTR, True)
    setattr(_parse_string_list, "_track2pclean_original", original)
    benchmark._parse_string_list = (
        _parse_string_list  # pylint: disable=protected-access
    )


__all__ = ["install_benchmark_string_list_validation"]
