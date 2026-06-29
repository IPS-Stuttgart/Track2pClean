from __future__ import annotations

import bayescatrack  # noqa: F401
import numpy as np
import pytest
from bayescatrack.experiments.track2p_benchmark import ProgressReporter


@pytest.mark.parametrize("enabled", ["false", "", 1, 0, None])
def test_progress_reporter_rejects_non_boolean_enabled(enabled: object) -> None:
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        ProgressReporter(1, enabled=enabled, label="demo")


@pytest.mark.parametrize("enabled", [True, False, np.bool_(True), np.bool_(False)])
def test_progress_reporter_accepts_boolean_enabled(enabled: object) -> None:
    reporter = ProgressReporter(2, enabled=enabled, label="demo")

    assert reporter.enabled is bool(enabled)
