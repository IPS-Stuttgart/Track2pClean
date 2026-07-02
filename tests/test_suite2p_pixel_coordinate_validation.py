from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import load_suite2p_plane


def _write_suite2p_stat(tmp_path, *, ypix: object, xpix: object) -> None:
    ypix_array = np.asarray(ypix)
    stat = np.asarray(
        [
            {
                "ypix": ypix_array,
                "xpix": np.asarray(xpix),
                "lam": np.ones(ypix_array.shape, dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)
    np.save(tmp_path / "ops.npy", {"Ly": 4, "Lx": 5})


def _object_array_token(value: object) -> np.ndarray:
    array = np.empty((1,), dtype=object)
    array[0] = value
    return array


@pytest.mark.parametrize(
    ("ypix", "xpix", "message"),
    [
        ([0.5], [0], "ypix"),
        ([np.nan], [0], "ypix"),
        ([True], [0], "ypix"),
        (["1"], [0], "ypix"),
        ([[0, 1]], [[2, 3]], "ypix.*one-dimensional"),
        ([0, 1], [[2, 3]], "xpix.*one-dimensional"),
        ([0], [1.5], "xpix"),
    ],
)
def test_load_suite2p_plane_rejects_malformed_pixel_coordinates(
    tmp_path,
    ypix: object,
    xpix: object,
    message: str,
) -> None:
    _write_suite2p_stat(tmp_path, ypix=ypix, xpix=xpix)

    with pytest.raises(ValueError, match=message):
        load_suite2p_plane(tmp_path, load_traces=False, load_spike_traces=False)


@pytest.mark.parametrize(
    "token",
    [
        bytearray(b"1"),
        memoryview(b"1"),
        np.asarray("1"),
        np.asarray(b"1"),
        np.asarray(True),
        np.asarray([1]),
    ],
)
def test_load_suite2p_plane_rejects_object_pixel_coordinate_tokens(
    tmp_path,
    token: object,
) -> None:
    _write_suite2p_stat(tmp_path, ypix=_object_array_token(token), xpix=[0])

    with pytest.raises(ValueError, match="ypix.*finite integer pixel coordinates"):
        load_suite2p_plane(tmp_path, load_traces=False, load_spike_traces=False)


def test_load_suite2p_plane_accepts_integer_like_float_pixel_coordinates(
    tmp_path,
) -> None:
    _write_suite2p_stat(tmp_path, ypix=[1.0], xpix=[2.0])

    plane = load_suite2p_plane(tmp_path, load_traces=False, load_spike_traces=False)

    assert plane.roi_masks.shape == (1, 4, 5)
    assert bool(plane.roi_masks[0, 1, 2])
