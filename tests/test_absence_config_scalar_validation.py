from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.absence_model import AbsenceModelConfig


@pytest.mark.parametrize(
    "bad_value",
    (
        "1.0",
        b"1.0",
        bytearray(b"1.0"),
        np.str_("1.0"),
        np.bytes_(b"1.0"),
        [1.0],
        np.asarray([1.0], dtype=float),
    ),
)
def test_absence_model_config_rejects_ambiguous_scalar_values(bad_value: object) -> None:
    with pytest.raises(ValueError, match="base_absence_cost"):
        AbsenceModelConfig(base_absence_cost=bad_value)


def test_absence_model_config_accepts_numpy_numeric_scalar() -> None:
    config = AbsenceModelConfig(base_absence_cost=np.asarray(1.25, dtype=float))

    assert config.base_absence_cost == pytest.approx(1.25)
