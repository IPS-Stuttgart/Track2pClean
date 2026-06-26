import numpy as np

from bayescatrack.core.bridge import CalciumPlaneData


def test_replaced_masks_drop_stale_fov_when_spatial_shape_changes():
    plane = CalciumPlaneData(
        roi_masks=np.ones((1, 2, 2), dtype=bool),
        traces=np.arange(3.0).reshape(1, 3),
        fov=np.ones((2, 2), dtype=float),
        cell_probabilities=np.asarray([0.8], dtype=float),
        roi_indices=np.asarray([7], dtype=int),
        roi_features={"radius": np.asarray([1.5], dtype=float)},
        source="suite2p",
        plane_name="plane0",
        ops={"Ly": 2, "Lx": 2},
    )

    replaced = plane.with_replaced_masks(np.ones((1, 3, 4), dtype=bool))

    assert replaced.roi_masks.shape == (1, 3, 4)
    assert replaced.fov is None
    assert replaced.traces.shape == (1, 3)
    assert replaced.cell_probabilities.tolist() == [0.8]
    assert replaced.roi_indices.tolist() == [7]
    assert replaced.roi_features["radius"].tolist() == [1.5]


def test_replaced_masks_preserve_fov_when_spatial_shape_matches():
    fov = np.ones((2, 2), dtype=float)
    plane = CalciumPlaneData(
        roi_masks=np.ones((1, 2, 2), dtype=bool),
        fov=fov,
    )

    replaced = plane.with_replaced_masks(np.zeros((1, 2, 2), dtype=bool))

    assert replaced.fov is fov
