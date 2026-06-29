from __future__ import annotations

import pytest
from bayescatrack._progress_reporter_validation import install_progress_reporter_validation
from bayescatrack.experiments.track2p_benchmark import ProgressReporter


def test_progress_reporter_rejects_text_enabled() -> None:
    install_progress_reporter_validation()

    with pytest.raises(ValueError, match="enabled must be a boolean"):
        ProgressReporter(1, enabled="value", label="demo")
