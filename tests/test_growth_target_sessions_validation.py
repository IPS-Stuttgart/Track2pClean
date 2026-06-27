from __future__ import annotations

import pytest

from bayescatrack.analysis.growth import _target_sessions


def test_growth_target_sessions_reject_duplicate_normalized_indices() -> None:
    with pytest.raises(ValueError, match="duplicate target_sessions"):
        _target_sessions(n_sessions=3, source_session=0, target_sessions=(1, "1.0"))
