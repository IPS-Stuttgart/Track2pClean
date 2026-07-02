from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import load_suite2p_plane


def _object_array(values):
    array = np.empty((len(values),), dtype=object)
    array[:] = values
    return array


def _write_suite2p_stat(
    plane_dir, lam: np.ndarray, *, coerce_lam_to_float: bool = True
) -> None:
    lam_values = np.asarray(lam, dtype=float) if coerce_lam_to_float else lam
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([0, 1], dtype=int),
                "xpix": np.asarray([0, 1], dtype=int),
                "lam": lam_values,
            }
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)


@pytest.mark.parametrize(
    "bad_lam",
    [
        np.asarray([1.0, np.nan], dtype=float),
        np.asarray([1.0, np.inf], dtype=float),
        np.asarray([1.0, -0.25], dtype=float),
    ],
)
def test_load_suite2p_plane_rejects_invalid_weighted_lam_values(tmp_path, bad_lam):
    _write_suite2p_stat(tmp_path, bad_lam)

    with pytest.raises(ValueError, match="lam values.*finite and non-negative"):
        load_suite2p_plane(
            tmp_path,
            weighted_masks=True,
            load_traces=False,
            load_spike_traces=False,
        )


@pytest.mark.parametrize(
    "bad_lam",
    [
        np.asarray(["1.0", "2.0"], dtype=np.str_),
        np.asarray([b"1.0", b"2.0"], dtype=np.bytes_),
        _object_array([bytearray(b"1"), bytearray(b"2")]),
    ],
)
def test_load_suite2p_plane_rejects_text_like_weighted_lam_values(tmp_path, bad_lam):
    _write_suite2p_stat(tmp_path, bad_lam, coerce_lam_to_float=False)

    with pytest.raises(ValueError, match="lam values.*finite and non-negative"):
        load_suite2p_plane(
            tmp_path,
            weighted_masks=True,
            load_traces=False,
            load_spike_traces=False,
        )


def test_load_suite2p_plane_accepts_valid_weighted_lam_values(tmp_path):
    _write_suite2p_stat(tmp_path, np.asarray([0.5, 1.25], dtype=float))

    plane = load_suite2p_plane(
        tmp_path,
        weighted_masks=True,
        load_traces=False,
        load_spike_traces=False,
    )

    assert plane.roi_masks.dtype.kind == "f"
    np.testing.assert_allclose(
        plane.roi_masks[0, np.asarray([0, 1]), np.asarray([0, 1])],
        np.asarray([0.5, 1.25], dtype=float),
    )
