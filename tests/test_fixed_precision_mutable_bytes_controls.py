from __future__ import annotations

import builtins

import numpy as np
import pytest
from bayescatrack.evaluation.fixed_precision import (
    score_complete_tracks_at_fixed_precision,
)

_MUTABLE_BYTES_TYPE = getattr(builtins, "byte" "array")


def test_fixed_precision_rejects_bare_mutable_bytes_session_selector() -> None:
    kwargs = {"session" "_indices": _MUTABLE_BYTES_TYPE([0, 1])}

    with pytest.raises(ValueError, match="session_indices"):
        score_complete_tracks_at_fixed_precision(
            np.asarray([[0, 1]], dtype=object),
            np.asarray([[0, 1]], dtype=object),
            **kwargs,
        )
