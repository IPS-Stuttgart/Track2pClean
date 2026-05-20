from __future__ import annotations

import numpy as np
import numpy.testing as npt
from bayescatrack import CalciumPlaneData, load_suite2p_plane


def _single_roi_plane(
    *,
    cell_probabilities: list[float] | None = None,
    roi_features: dict[str, np.ndarray] | None = None,
) -> CalciumPlaneData:
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[0, 1:3, 1:3] = True

    kwargs = {
        "roi_masks": mask,
        "roi_indices": np.asarray([0], dtype=int),
        "source": "synthetic",
    }
    if cell_probabilities is not None:
        kwargs["cell_probabilities"] = np.asarray(cell_probabilities, dtype=float)
    if roi_features is not None:
        kwargs["roi_features"] = roi_features
    return CalciumPlaneData(**kwargs)


def test_load_suite2p_plane_without_iscell_leaves_cell_probabilities_unknown(
    tmp_path,
):
    plane_dir = tmp_path / "plane0"
    plane_dir.mkdir()

    stat = np.empty(1, dtype=object)
    stat[0] = {
        "ypix": np.asarray([0, 1], dtype=int),
        "xpix": np.asarray([0, 1], dtype=int),
        "lam": np.ones(2, dtype=float),
    }
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "ops.npy", {"Ly": 2, "Lx": 2})

    plane = load_suite2p_plane(plane_dir)

    assert plane.cell_probabilities is None


def test_cell_probability_cost_ignores_unknown_values():
    reference = _single_roi_plane(cell_probabilities=[np.nan])
    measurement = _single_roi_plane(cell_probabilities=[0.9])

    cost_matrix, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=0.0,
        cell_probability_weight=1.0,
        return_components=True,
    )

    npt.assert_allclose(components["cell_probability_available"], np.array([[0.0]]))
    npt.assert_allclose(components["cell_probability_cost"], np.array([[0.0]]))
    npt.assert_allclose(cost_matrix, np.array([[0.0]]))


def test_roi_feature_distance_uses_per_pair_available_dimensions():
    reference = _single_roi_plane(
        roi_features={
            "radius": np.asarray([1.0]),
            "skew": np.asarray([np.nan]),
        }
    )
    measurement = _single_roi_plane(
        roi_features={
            "radius": np.asarray([3.0]),
            "skew": np.asarray([7.0]),
        }
    )

    _, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=1.0,
        feature_names=("radius", "skew"),
        return_components=True,
    )

    npt.assert_allclose(components["roi_feature_cost"], np.array([[2.0]]))
