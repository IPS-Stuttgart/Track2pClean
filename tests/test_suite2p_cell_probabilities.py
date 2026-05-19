from __future__ import annotations

import numpy as np

from bayescatrack.core.bridge import CalciumPlaneData, load_suite2p_plane


def _minimal_masks() -> np.ndarray:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    return masks


def _write_minimal_suite2p_plane(plane_dir) -> None:
    stat = np.array(
        [
            {
                "ypix": np.array([0, 0, 1, 1]),
                "xpix": np.array([0, 1, 0, 1]),
                "lam": np.ones(4),
            },
            {
                "ypix": np.array([2, 2, 3, 3]),
                "xpix": np.array([2, 3, 2, 3]),
                "lam": np.ones(4),
            },
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)
    np.save(
        plane_dir / "ops.npy",
        {"Ly": 4, "Lx": 4, "meanImg": np.zeros((4, 4), dtype=float)},
    )


def test_load_suite2p_plane_without_iscell_omits_cell_probabilities(tmp_path):
    _write_minimal_suite2p_plane(tmp_path)

    plane = load_suite2p_plane(tmp_path)

    assert plane.n_rois == 2
    assert plane.cell_probabilities is None


def test_pairwise_cell_probability_cost_ignores_nonfinite_entries():
    reference = CalciumPlaneData(
        roi_masks=_minimal_masks(),
        cell_probabilities=np.array([0.5, np.nan]),
    )
    measurement = CalciumPlaneData(
        roi_masks=_minimal_masks(),
        cell_probabilities=np.array([np.nan, 0.25]),
    )

    costs, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=0.0,
        cell_probability_weight=1.0,
        return_components=True,
    )

    assert np.all(np.isfinite(costs))
    assert np.array_equal(costs, components["cell_probability_cost"])
    assert costs[0, 0] == 0.0
    assert costs[1, 0] == 0.0
    assert costs[1, 1] == 0.0
    assert costs[0, 1] > 0.0
