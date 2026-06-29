import numpy as np
import pytest

from bayescatrack import load_suite2p_plane


class _OverflowingDimension:
    def __float__(self):
        raise OverflowError("cannot fit")


def _write_minimal_suite2p_plane(plane_dir, *, ly=2, lx=2):
    plane_dir.mkdir(parents=True)
    stat = np.empty(1, dtype=object)
    stat[0] = {"ypix": np.array([0]), "xpix": np.array([0]), "lam": np.array([1.0])}
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "ops.npy", {"Ly": ly, "Lx": lx, "meanImg": np.ones((2, 2))})
    np.save(plane_dir / "iscell.npy", np.array([[1.0, 0.9]], dtype=float))


@pytest.mark.parametrize(
    "bad_dimension",
    [
        "2",
        b"2",
        np.array("2"),
        np.array([2]),
    ],
)
def test_load_suite2p_plane_rejects_ambiguous_ops_image_dimensions(
    tmp_path, bad_dimension
):
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir, ly=bad_dimension)

    with pytest.raises(
        ValueError,
        match="Suite2p ops Ly must be a positive integer image dimension",
    ):
        load_suite2p_plane(plane_dir)


def test_load_suite2p_plane_normalizes_overflowing_ops_image_dimension(tmp_path):
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir, lx=_OverflowingDimension())

    with pytest.raises(
        ValueError,
        match="Suite2p ops Lx must be a positive integer image dimension",
    ):
        load_suite2p_plane(plane_dir)
