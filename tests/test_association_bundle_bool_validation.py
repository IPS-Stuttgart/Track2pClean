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


def _session_with_rois(name: str, roi_indices: list[int]) -> Track2pSession:
    return Track2pSession(
        session_dir=Path(name),
        session_name=name,
        session_date=None,
        plane_data=_plane_with_rois(roi_indices),
    )


def _plane_with_rois(roi_indices: list[int]) -> CalciumPlaneData:
    masks = np.zeros((len(roi_indices), 8, 8), dtype=bool)
    for row, roi_index in enumerate(roi_indices):
        masks[row, 1 + row, 1 + (roi_index % 5)] = True
    return CalciumPlaneData(
        roi_masks=masks,
        roi_indices=np.asarray(roi_indices, dtype=int),
        source="association_plane_validation",
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


def test_session_pair_association_bundle_rejects_wrong_transformed_roi_count() -> None:
    reference = _session_with_rois("2024-01-01", [0])
    measurement = _session_with_rois("2024-01-02", [10, 11])
    transformed = _plane_with_rois([10])

    with pytest.raises(ValueError, match="preserve the measurement session ROI count"):
        build_session_pair_association_bundle(
            reference,
            measurement,
            measurement_plane_in_reference_frame=transformed,
        )


def test_session_pair_association_bundle_rejects_wrong_transformed_roi_ids() -> None:
    reference = _session_with_rois("2024-01-01", [0])
    measurement = _session_with_rois("2024-01-02", [10, 11])
    transformed = _plane_with_rois([10, 12])

    with pytest.raises(ValueError, match="preserve the measurement session ROI identities"):
        build_session_pair_association_bundle(
            reference,
            measurement,
            measurement_plane_in_reference_frame=transformed,
        )


def test_session_pair_association_bundle_accepts_permuted_transformed_roi_ids() -> None:
    reference = _session_with_rois("2024-01-01", [0])
    measurement = _session_with_rois("2024-01-02", [10, 11])
    transformed = _plane_with_rois([11, 10])

    bundle = build_session_pair_association_bundle(
        reference,
        measurement,
        measurement_plane_in_reference_frame=transformed,
    )

    np.testing.assert_array_equal(bundle.measurement_roi_indices, np.asarray([11, 10]))
