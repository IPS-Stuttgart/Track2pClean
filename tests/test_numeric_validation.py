from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association._numeric_validation import (
    finite_positive_float,
    integer,
    nonnegative_integer,
    positive_integer,
    probability,
    validated_numeric_float,
)


class _BadIndexWithFloatFallback:
    def __float__(self) -> float:
        return 1.0


def _dunder(name: str) -> str:
    return "_" * 2 + name + "_" * 2


def _raise_bad_index(_self: object) -> int:
    raise OverflowError("too large")


setattr(_BadIndexWithFloatFallback, _dunder("index"), _raise_bad_index)


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
    "value",
    [
        "0.5",
        b"0.5",
        bytearray(b"0.5"),
        np.str_("0.5"),
        np.bytes_(b"0.5"),
        np.asarray("0.5"),
        np.asarray(b"0.5"),
    ],
)
def test_validated_numeric_float_rejects_string_like_numeric_values(value):
    with pytest.raises(ValueError, match="control must be finite"):
        validated_numeric_float(value, name="control")


@pytest.mark.parametrize(
    "value",
    [
        [1.0],
        (1.0,),
        np.asarray([1.0]),
        np.asarray([[1.0]]),
    ],
)
def test_validated_numeric_float_rejects_nonscalar_array_like_values(value):
    with pytest.raises(ValueError, match="control must be finite"):
        validated_numeric_float(value, name="control")


@pytest.mark.parametrize(
    "value",
    [
        np.asarray(True),
        np.asarray(False),
        np.asarray([True]),
        np.asarray([False]),
    ],
)
def test_validated_numeric_float_rejects_boolean_array_values(value):
    with pytest.raises(ValueError, match="control must be finite"):
        validated_numeric_float(value, name="control")


@pytest.mark.parametrize("value", [np.asarray(1.25), np.asarray(2)])
def test_validated_numeric_float_accepts_zero_dimensional_numeric_arrays(value):
    assert validated_numeric_float(value, name="control") == float(value.item())


@pytest.mark.parametrize(
    "validator",
    [finite_positive_float, probability, nonnegative_integer],
)
def test_derived_numeric_validators_preserve_named_value_errors(validator):
    with pytest.raises(ValueError, match="custom_name"):
        validator("not-a-number", name="custom_name")


@pytest.mark.parametrize(
    ("validator", "message"),
    [
        (integer, "control must be an integer"),
        (positive_integer, "control must be a positive integer"),
        (nonnegative_integer, "control must be a non-negative integer"),
    ],
)
def test_integer_validators_normalize_bad_index_errors_before_float_fallback(
    validator, message: str
):
    with pytest.raises(ValueError, match=message):
        validator(_BadIndexWithFloatFallback(), name="control")
