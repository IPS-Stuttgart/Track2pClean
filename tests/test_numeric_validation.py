from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association._numeric_validation import (
    finite_positive_float,
    nonnegative_integer,
    probability,
    validated_numeric_float,
)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not-a-number",
        object(),
        1.0 + 2.0j,
        np.asarray([1.0, 2.0]),
        pytest.param(10**10000, id="overflowing-int"),
    ],
)
def test_validated_numeric_float_rejects_uncoercible_values_as_value_error(value):
    with pytest.raises(ValueError, match="control must be finite"):
        validated_numeric_float(value, name="control")


@pytest.mark.parametrize(
    "validator",
    [finite_positive_float, probability, nonnegative_integer],
)
def test_derived_numeric_validators_preserve_named_value_errors(validator):
    with pytest.raises(ValueError, match="custom_name"):
        validator("not-a-number", name="custom_name")
