from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import bayescatrack
import numpy as np
import pytest
from bayescatrack.core.bridge import (
    CalciumPlaneData,
    Track2pSession,
    build_consecutive_session_association_bundles,
    build_session_pair_association_bundle,
)


def _session(name: str) -> Track2pSession:
    mask = np.zeros((1, 5, 5), dtype=bool)
    mask[0, 2, 2] = True
    return Track2pSession(
        session_dir=Path(name),
        session_name=name,
        session_date=None,
        plane_data=CalciumPlaneData(
            roi_masks=mask,
            roi_indices=np.asarray([0], dtype=int),
            source="association_bundle_bool_validation",
        ),
    )


@pytest.mark.parametrize(
    "builder",
    [
        build_session_pair_association_bundle,
        bayescatrack.build_session_pair_association_bundle,
    ],
)
@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("weighted_centroids", 1),
        ("return_pairwise_components", "false"),
    ],
)
def test_session_pair_association_bundle_rejects_non_boolean_controls(
    builder: Callable[..., Any],
    keyword: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=f"{keyword} must be a boolean"):
        builder(_session("2024-01-01"), _session("2024-01-02"), **{keyword: value})


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("weighted_centroids", "false"),
        ("return_pairwise_components", 0),
    ],
)
def test_consecutive_association_bundles_rejects_non_boolean_controls(
    keyword: str,
    value: object,
) -> None:
    sessions = [_session("2024-01-01"), _session("2024-01-02")]

    with pytest.raises(ValueError, match=f"{keyword} must be a boolean"):
        build_consecutive_session_association_bundles(sessions, **{keyword: value})


def test_association_bundle_builders_accept_numpy_booleans() -> None:
    reference = _session("2024-01-01")
    measurement = _session("2024-01-02")

    bundle = build_session_pair_association_bundle(
        reference,
        measurement,
        weighted_centroids=np.bool_(False),
        return_pairwise_components=np.bool_(False),
    )

    assert bundle.pairwise_components == {}


def test_pairwise_cost_return_components_key_cannot_leak_tuple_cost_matrix() -> None:
    bundle = build_session_pair_association_bundle(
        _session("2024-01-01"),
        _session("2024-01-02"),
        return_pairwise_components=False,
        pairwise_cost_kwargs={"return_components": True},
    )

    assert isinstance(bundle.pairwise_cost_matrix, np.ndarray)
    assert bundle.pairwise_cost_matrix.shape == (1, 1)
    assert bundle.pairwise_components == {}
