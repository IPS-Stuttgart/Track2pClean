from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    apply_dynamic_edge_priors,
)


@pytest.mark.parametrize(
    "session_gap",
    [b"2", bytearray(b"2"), memoryview(b"2"), np.bytes_(b"2")],
)
def test_dynamic_edge_prior_rejects_binary_session_gap_controls(session_gap) -> None:
    with pytest.raises(ValueError, match="session_gap"):
        apply_dynamic_edge_priors(
            np.asarray([[1.0]], dtype=float),
            {},
            session_gap=session_gap,
            config=DynamicEdgePriorConfig(session_gap_weight=0.25),
        )


def test_dynamic_edge_prior_rejects_numeric_string_session_gap() -> None:
    with pytest.raises(ValueError, match="session_gap"):
        apply_dynamic_edge_priors(
            np.asarray([[1.0]], dtype=float),
            {},
            session_gap="3",
            config=DynamicEdgePriorConfig(session_gap_weight=0.25),
        )


def test_dynamic_edge_prior_preserves_numeric_session_gap() -> None:
    adjusted = apply_dynamic_edge_priors(
        np.asarray([[1.0]], dtype=float),
        {},
        session_gap=3,
        config=DynamicEdgePriorConfig(session_gap_weight=0.25),
    )

    assert adjusted[0, 0] == pytest.approx(1.5)
