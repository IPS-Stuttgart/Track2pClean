from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from bayescatrack.association.adaptive_priors import apply_adaptive_edge_priors


@dataclass(frozen=True)
class _Plane:
    n_rois: int


@dataclass(frozen=True)
class _Session:
    plane_data: _Plane


def test_adaptive_edge_priors_reject_memoryview_edge_key() -> None:
    sessions = (_Session(_Plane(0)), _Session(_Plane(0)))
    costs = {memoryview(b"\x00\x01"): np.zeros((0, 0), dtype=float)}

    with pytest.raises(ValueError, match="session-edge pairs"):
        apply_adaptive_edge_priors(costs, sessions)
