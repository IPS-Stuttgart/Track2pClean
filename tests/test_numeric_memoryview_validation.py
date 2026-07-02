from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association._numeric_validation import validated_numeric_float


def _object_scalar(value: object) -> np.ndarray:
    array = np.empty((), dtype=object)
    array[()] = value
    return array


def test_validated_numeric_float_rejects_memoryview_object_scalar() -> None:
    value = _object_scalar(memoryview("0.5".encode()))

    with pytest.raises(ValueError, match="control must be finite"):
        validated_numeric_float(value, name="control")
