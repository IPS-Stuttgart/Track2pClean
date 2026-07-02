from __future__ import annotations

from importlib import reload as reload_module

import numpy as np
import pytest
import bayescatrack.association as association
from bayescatrack.association import (
    _dynamic_edge_prior_validation as dynamic_edge_prior_validation,
)
from bayescatrack.association import dynamic_edge_priors
from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    apply_dynamic_edge_priors,
)


def _session_gap_validation_wrapper_count() -> int:
    marker = dynamic_edge_prior_validation._SESSION_GAP_PATCH_MARKER
    original_attributes = (
        dynamic_edge_prior_validation._SESSION_GAP_ORIGINAL_ATTR,
        "_bayescatrack_original",
    )
    current = dynamic_edge_priors.apply_dynamic_edge_priors
    seen: set[int] = set()
    count = 0
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            break
        if getattr(current, marker, False):
            count += 1
        seen.add(current_id)
        current = next(
            (
                getattr(current, attribute_name)
                for attribute_name in original_attributes
                if getattr(current, attribute_name, None) is not None
            ),
            None,
        )
    return count


def test_dynamic_edge_prior_rejects_nonfinite_edge_quality_bias():
    for invalid_bias in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="edge_quality_bias must be finite"):
            DynamicEdgePriorConfig(edge_quality_bias=invalid_bias)


def test_dynamic_edge_prior_allows_finite_negative_edge_quality_bias():
    costs = np.asarray([[3.0, 4.0]], dtype=float)

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(edge_quality_bias=-0.5),
    )

    np.testing.assert_allclose(adjusted, np.asarray([[2.5, 3.5]], dtype=float))


def test_dynamic_edge_prior_rejects_invalid_session_gap_when_gap_weighted():
    costs = np.asarray([[3.0, 4.0]], dtype=float)

    for invalid_gap in (0, -1, True, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="session_gap must be a finite value"):
            apply_dynamic_edge_priors(
                costs,
                {},
                session_gap=invalid_gap,
                config=DynamicEdgePriorConfig(session_gap_weight=0.25),
            )


def test_dynamic_edge_prior_rejects_text_like_session_gap_when_gap_weighted():
    costs = np.asarray([[3.0, 4.0]], dtype=float)

    for invalid_gap in (
        "2",
        b"2",
        bytearray(b"2"),
        np.str_("2"),
        np.bytes_(b"2"),
        np.asarray("2"),
    ):
        with pytest.raises(ValueError, match="session_gap must be a finite value"):
            apply_dynamic_edge_priors(
                costs,
                {},
                session_gap=invalid_gap,
                config=DynamicEdgePriorConfig(session_gap_weight=0.25),
            )


def test_dynamic_edge_prior_rejects_array_valued_session_gap_when_gap_weighted():
    costs = np.asarray([[3.0, 4.0]], dtype=float)

    for invalid_gap in (
        np.asarray([2]),
        np.asarray([[2]]),
        np.asarray([2.0]),
    ):
        with pytest.raises(ValueError, match="session_gap must be a finite value"):
            apply_dynamic_edge_priors(
                costs,
                {},
                session_gap=invalid_gap,
                config=DynamicEdgePriorConfig(session_gap_weight=0.25),
            )


def test_dynamic_edge_prior_session_gap_validation_keeps_valid_offsets():
    costs = np.asarray([[3.0, 4.0]], dtype=float)

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=3,
        config=DynamicEdgePriorConfig(session_gap_weight=0.25),
    )

    np.testing.assert_allclose(adjusted, np.asarray([[3.5, 4.5]], dtype=float))


def test_dynamic_edge_prior_session_gap_validation_keeps_numpy_scalar_offsets():
    costs = np.asarray([[3.0, 4.0]], dtype=float)

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=np.asarray(3),
        config=DynamicEdgePriorConfig(session_gap_weight=0.25),
    )

    np.testing.assert_allclose(adjusted, np.asarray([[3.5, 4.5]], dtype=float))


def test_dynamic_edge_prior_session_gap_validation_is_reload_idempotent():
    reload_module(association)

    assert _session_gap_validation_wrapper_count() == 1
