from __future__ import annotations

import pytest

from bayescatrack._cli_exit_code_validation import _coerce_exit_code


class BadIndex:
    def __index__(self) -> int:
        raise ArithmeticError("bad exit code")


def test_exit_code_coercion_wraps_arithmetic_index_errors() -> None:
    with pytest.raises(TypeError, match="integer exit code"):
        _coerce_exit_code(BadIndex())
