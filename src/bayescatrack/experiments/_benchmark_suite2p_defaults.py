"""Align programmatic Track2p benchmark Suite2p defaults with CLI defaults."""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

_PATCH_MARKER = "_track2pclean_benchmark_suite2p_defaults_patch"


def install_benchmark_suite2p_defaults() -> None:
    """Patch Track2pBenchmarkConfig defaults idempotently."""

    from bayescatrack.experiments import track2p_benchmark as benchmark

    config_cls = benchmark.Track2pBenchmarkConfig
    if getattr(config_cls, _PATCH_MARKER, False):
        return

    _replace_init_default(config_cls, "include_non_cells", True)
    _replace_dataclass_field_default(config_cls, "include_non_cells", True)
    setattr(config_cls, _PATCH_MARKER, True)


def _replace_init_default(config_cls: type[Any], field_name: str, value: Any) -> None:
    init = config_cls.__init__
    defaults = init.__defaults__
    if defaults is None:
        raise RuntimeError(f"{config_cls.__name__}.__init__ has no default tuple")

    defaulted_fields = tuple(
        field
        for field in fields(config_cls)
        if field.init
        and (field.default is not MISSING or field.default_factory is not MISSING)
    )
    default_names = tuple(field.name for field in defaulted_fields)
    try:
        default_index = default_names.index(field_name)
    except ValueError as exc:  # pragma: no cover - defensive wiring guard
        raise RuntimeError(
            f"{config_cls.__name__} has no defaulted field {field_name!r}"
        ) from exc

    if len(defaults) != len(default_names):  # pragma: no cover - defensive wiring guard
        raise RuntimeError(
            f"{config_cls.__name__} default count does not match dataclass fields"
        )

    patched_defaults = list(defaults)
    patched_defaults[default_index] = value
    init.__defaults__ = tuple(patched_defaults)


def _replace_dataclass_field_default(
    config_cls: type[Any], field_name: str, value: Any
) -> None:
    dataclass_fields = getattr(config_cls, "__dataclass_fields__")
    dataclass_fields[field_name].default = value


__all__ = ["install_benchmark_suite2p_defaults"]
