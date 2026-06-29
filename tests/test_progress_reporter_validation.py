from __future__ import annotations

import pytest
from bayescatrack.experiments.track2p_benchmark import ProgressReporter


def test_progress_reporter_rejects_text_enabled() -> None:
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        ProgressReporter(1, enabled="yes", label="demo")
