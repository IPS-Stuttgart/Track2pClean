from __future__ import annotations

import pytest
from bayescatrack.analysis.growth import _target_sessions


def test_growth_target_sessions_rejects_memoryview_selector():
    with pytest.raises(ValueError, match="target_sessions.*string-like"):
        _target_sessions(
            n_sessions=100,
            source_session=2,
            target_sessions=memoryview(b"10"),
        )
