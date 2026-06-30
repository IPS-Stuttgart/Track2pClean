from __future__ import annotations

import pytest
from bayescatrack.evaluation import calibration_diagnostics as diagnostics


def test_numeric_text_threshold_value_is_rejected() -> None:
    value = "".join(("0", ".", "5"))
    with pytest.raises(ValueError, match="thresholds must be finite numeric"):
        diagnostics._validate_probability_threshold(value)
