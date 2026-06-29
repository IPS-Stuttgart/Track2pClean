from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import load_suite2p_plane, load_track2p_subject

_STRING_LIKE_PROBABILITY_VALUES = [
    "0.5",
    b"0.5",
    bytearray(b"0.5"),
    np.str_("0.5"),
    np.bytes_(b"0.5"),
    np.array("0.5"),
]


class _ArithmeticFloat:
    def __float__(self) -> float:
        raise ArithmeticError("bad numeric conversion")


def _write_minimal_suite2p_plane(plane_dir):
    plane_dir.mkdir(parents=True)
    stat = np.empty(1, dtype=object)
    stat[0] = {
        "ypix": np.array([0]),
        "xpix": np.array([0]),
        "lam": np.array([1.0]),
        "overlap": np.array([False]),
    }
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "iscell.npy", np.array([[1.0, 0.75]], dtype=float))
    np.save(
        plane_dir / "ops.npy",
        {"Ly": 2, "Lx": 2, "meanImg": np.ones((2, 2), dtype=float)},
        allow_pickle=True,
    )


@pytest.mark.parametrize("bad_value", [np.array([0.5]), np.array([[0.5]])])
def test_load_suite2p_plane_rejects_array_probability_threshold(tmp_path, bad_value):
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir)

    with pytest.raises(
        ValueError, match="cell_probability_threshold must be a finite probability"
    ):
        load_suite2p_plane(plane_dir, cell_probability_threshold=bad_value)


@pytest.mark.parametrize("bad_value", _STRING_LIKE_PROBABILITY_VALUES)
def test_load_suite2p_plane_rejects_string_like_probability_threshold(
    tmp_path, bad_value
):
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir)

    with pytest.raises(
        ValueError, match="cell_probability_threshold must be a finite probability"
    ):
        load_suite2p_plane(plane_dir, cell_probability_threshold=bad_value)


@pytest.mark.parametrize("bad_value", _STRING_LIKE_PROBABILITY_VALUES)
def test_load_track2p_subject_rejects_string_like_probability_threshold(
    tmp_path, bad_value
):
    subject_dir = tmp_path / "subject"
    plane_dir = subject_dir / "2024-01-01_session" / "suite2p" / "plane0"
    _write_minimal_suite2p_plane(plane_dir)

    with pytest.raises(
        ValueError, match="cell_probability_threshold must be a finite probability"
    ):
        load_track2p_subject(
            subject_dir,
            input_format="suite2p",
            plane_name="plane0",
            cell_probability_threshold=bad_value,
            load_traces=False,
            load_spike_traces=False,
        )


def test_load_suite2p_plane_accepts_zero_dimensional_probability_threshold(tmp_path):
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir)

    plane = load_suite2p_plane(plane_dir, cell_probability_threshold=np.array(0.5))

    assert plane.n_rois == 1


def test_load_suite2p_plane_wraps_arithmetic_probability_threshold(tmp_path):
    plane_dir = tmp_path / "plane0"
    _write_minimal_suite2p_plane(plane_dir)

    with pytest.raises(
        ValueError, match="cell_probability_threshold must be a finite probability"
    ):
        load_suite2p_plane(
            plane_dir,
            cell_probability_threshold=_ArithmeticFloat(),
        )
