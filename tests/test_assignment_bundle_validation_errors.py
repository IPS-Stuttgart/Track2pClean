from __future__ import annotations

import pytest
from bayescatrack._assignment_bundle_validation import _normalize_roi_index_array


class _IndexOverflow:
    def __index__(self) -> int:
        raise OverflowError("index conversion overflowed")


class _IndexValueError:
    def __index__(self) -> int:
        raise ValueError("index conversion failed")


@pytest.mark.parametrize("bad_value", [_IndexOverflow(), _IndexValueError()])
def test_assignment_bundle_roi_indices_normalize_index_protocol_errors(
    bad_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="bundle.reference_roi_indices must contain integer ROI indices",
    ):
        _normalize_roi_index_array([bad_value], "bundle.reference_roi_indices")
