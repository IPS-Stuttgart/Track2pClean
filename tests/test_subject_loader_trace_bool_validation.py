from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import export_subject_to_npz, load_track2p_subject, summarize_subject


_TRACE_FLAG_NAMES = ("load_traces", "load_spike_traces", "load_neuropil_traces")
_SUBJECT_ENTRYPOINTS = [
    (summarize_subject, ()),
    (export_subject_to_npz, ("subject.npz",)),
]


def _write_minimal_suite2p_plane(plane_dir):
    plane_dir.mkdir(parents=True)
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([0], dtype=int),
                "xpix": np.asarray([0], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)


def _entrypoint_call_args(tmp_path, extra_args):
    return [
        tmp_path,
        *(
            tmp_path / output_path if isinstance(output_path, str) else output_path
            for output_path in extra_args
        ),
    ]


@pytest.mark.parametrize("flag_name", _TRACE_FLAG_NAMES)
def test_load_track2p_subject_accepts_numpy_bool_trace_controls(tmp_path, flag_name):
    _write_minimal_suite2p_plane(tmp_path / "2024-01-01" / "suite2p" / "plane0")

    subject = load_track2p_subject(tmp_path, **{flag_name: np.bool_(False)})

    assert len(subject.sessions) == 1


@pytest.mark.parametrize("flag_name", _TRACE_FLAG_NAMES)
def test_load_track2p_subject_rejects_array_bool_trace_controls(tmp_path, flag_name):
    _write_minimal_suite2p_plane(tmp_path / "2024-01-01" / "suite2p" / "plane0")

    with pytest.raises(ValueError, match=flag_name):
        load_track2p_subject(tmp_path, **{flag_name: np.array(False)})


@pytest.mark.parametrize(("entrypoint", "extra_args"), _SUBJECT_ENTRYPOINTS)
@pytest.mark.parametrize("flag_name", _TRACE_FLAG_NAMES)
def test_subject_summary_and_export_accept_numpy_bool_trace_controls(
    tmp_path,
    entrypoint,
    extra_args,
    flag_name,
):
    _write_minimal_suite2p_plane(tmp_path / "2024-01-01" / "suite2p" / "plane0")
    call_args = _entrypoint_call_args(tmp_path, extra_args)

    result = entrypoint(*call_args, **{flag_name: np.bool_(False)})

    assert result["n_sessions"] == 1


@pytest.mark.parametrize(("entrypoint", "extra_args"), _SUBJECT_ENTRYPOINTS)
@pytest.mark.parametrize("flag_name", _TRACE_FLAG_NAMES)
def test_subject_summary_and_export_reject_array_bool_trace_controls(
    tmp_path,
    entrypoint,
    extra_args,
    flag_name,
):
    _write_minimal_suite2p_plane(tmp_path / "2024-01-01" / "suite2p" / "plane0")
    call_args = _entrypoint_call_args(tmp_path, extra_args)

    with pytest.raises(ValueError, match=flag_name):
        entrypoint(*call_args, **{flag_name: np.array(False)})


def test_export_subject_npz_allows_non_suite2p_trace_flags_for_npy_inputs(tmp_path):
    plane_dir = tmp_path / "2024-01-01" / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True)
    np.save(plane_dir / "rois.npy", np.ones((1, 1, 1), dtype=bool))
    np.save(plane_dir / "F.npy", np.ones((1, 3), dtype=float))
    np.save(plane_dir / "fov.npy", np.ones((1, 1), dtype=float))

    summary = export_subject_to_npz(
        tmp_path,
        tmp_path / "subject.npz",
        input_format="npy",
        load_traces=np.bool_(False),
    )

    assert summary["n_sessions"] == 1
