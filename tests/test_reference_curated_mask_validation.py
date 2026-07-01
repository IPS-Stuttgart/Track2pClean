from __future__ import annotations

from fractions import Fraction

import pytest
from bayescatrack.reference import Track2pReference


def test_reference_curated_mask_rejects_overflowing_numeric_value() -> None:
    overflowing_value = Fraction(10**10000, 1)

    with pytest.raises(
        ValueError,
        match="curated_mask must contain only boolean or 0/1 values",
    ):
        Track2pReference(
            ("2024-05-01_a",),
            [[0]],
            curated_mask=[overflowing_value],
        )
