from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from bayescatrack._cli_exit_code_validation import _coerce_exit_code

ROOT = Path(__file__).resolve().parents[1]


class BadIndex:
    def __index__(self) -> int:
        raise ArithmeticError("bad exit code")


def _load_raw_track2pclean_cli_module() -> ModuleType:
    module_path = ROOT / "src" / "track2pclean" / "_cli.py"
    spec = importlib.util.spec_from_file_location(
        "tests._raw_track2pclean_cli_for_exit_code", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exit_code_coercion_wraps_arithmetic_index_errors() -> None:
    with pytest.raises(TypeError, match="integer exit code"):
        _coerce_exit_code(BadIndex())


def test_track2pclean_native_exit_code_coercion_wraps_arithmetic_index_errors() -> None:
    module = _load_raw_track2pclean_cli_module()
    coerce_exit_code = getattr(module, "_coerce_exit_code")

    with pytest.raises(TypeError, match="integer exit code"):
        coerce_exit_code(BadIndex())
