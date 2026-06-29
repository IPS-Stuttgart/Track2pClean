from __future__ import annotations

import numpy as np
import pytest
from bayescatrack._progress_reporter_validation import install_progress_reporter_validation
from bayescatrack.experiments.track2p_benchmark import ProgressReporter


def test_progress_reporter_rejects_text_enabled() -> None:
    install_progress_reporter_validation()

    with pytest.raises(ValueError, match="enabled must be a boolean"):
        ProgressReporter(1, enabled="value", label="demo")


@pytest.mark.parametrize(
    "invalid_total",
    [
        0,
        -1,
        True,
        "3",
        b"3",
        1.5,
        np.nan,
        np.array([1, 2]),
    ],
)
def test_progress_reporter_rejects_invalid_total(invalid_total: object) -> None:
    install_progress_reporter_validation()

    with pytest.raises(ValueError, match="total must be a positive integer"):
        ProgressReporter(invalid_total, enabled=True, label="demo")


def test_progress_reporter_accepts_numpy_integer_total() -> None:
    install_progress_reporter_validation()

    reporter = ProgressReporter(np.int64(2), enabled=False, label="demo")

    assert reporter.total == 2