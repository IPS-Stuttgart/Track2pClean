from __future__ import annotations

import pytest
from bayescatrack.association.absence_model import AbsenceModelConfig


class _OverflowingFloat:
    def __float__(self) -> float:
        raise OverflowError("numeric adapter overflow")


def test_absence_model_config_rejects_overflowing_numeric_adapter() -> None:
    with pytest.raises(ValueError, match="base_absence_cost"):
        AbsenceModelConfig(base_absence_cost=_OverflowingFloat())
