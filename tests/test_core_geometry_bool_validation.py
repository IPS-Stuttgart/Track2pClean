from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.core.bridge import CalciumPlaneData


def _weighted_plane() -> CalciumPlaneData:
    masks = np.zeros((1, 3, 3), dtype=float)
    masks[0, 0, 0] = 1.0
    masks[0, 0, 2] = 3.0
    return CalciumPlaneData(masks)


@pytest.mark.parametrize(
    ("method_name", "kwargs", "message"),
    [
        ("roi_areas", {"weighted": "false"}, "weighted"),
        ("centroids", {"weighted": "false"}, "weighted"),
        ("position_covariances", {"weighted": bytearray(b"0")}, "weighted"),
        ("to_measurement_matrix", {"weighted": object()}, "weighted"),
        (
            "to_constant_velocity_state_moments",
            {"weighted": np.array(False)},
            "weighted",
        ),
        ("to_export_dict", {"weighted": "true"}, "weighted"),
        ("to_export_dict", {"include_masks": "false"}, "include_masks"),
    ],
)
def test_geometry_boolean_controls_reject_ambiguous_keyword_values(
    method_name: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    plane = _weighted_plane()

    with pytest.raises(ValueError, match=message):
        getattr(plane, method_name)(**kwargs)


def test_centroids_rejects_ambiguous_positional_weighted() -> None:
    plane = _weighted_plane()

    with pytest.raises(ValueError, match="weighted"):
        plane.centroids("xy", np.array(False))


def test_pairwise_centroid_distances_rejects_ambiguous_weighted() -> None:
    plane = _weighted_plane()

    with pytest.raises(ValueError, match="weighted"):
        plane.pairwise_centroid_distances(plane, weighted=1)


def test_geometry_boolean_controls_accept_numpy_bool_scalars() -> None:
    plane = _weighted_plane()

    np.testing.assert_allclose(
        plane.roi_areas(weighted=np.bool_(True)), np.asarray([4.0])
    )
    np.testing.assert_allclose(
        plane.roi_areas(weighted=np.bool_(False)), np.asarray([2.0])
    )
