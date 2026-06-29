from __future__ import annotations

import pytest
from bayescatrack.core import bridge as bridge_module
from track2pclean import export_subject_to_npz as write_subject_npz
from track2pclean import summarize_subject


def _make_date_session(subject_dir):
    session_dir = subject_dir / "2024-01-01"
    session_dir.mkdir()
    return session_dir


def test_track2pclean_summarize_subject_strict_missing_plane_errors(tmp_path):
    _make_date_session(tmp_path)

    with pytest.raises(FileNotFoundError, match="neither"):
        summarize_subject(tmp_path, strict=True)


def test_track2pclean_export_subject_strict_missing_plane_errors_before_write(tmp_path):
    _make_date_session(tmp_path)
    output_path = tmp_path / "subject.npz"

    with pytest.raises(FileNotFoundError, match="neither"):
        write_subject_npz(tmp_path, output_path, strict=True)

    assert not output_path.exists()


def test_track2pclean_summary_and_export_parsers_accept_strict(tmp_path):
    parser = bridge_module._bridge_impl._build_arg_parser()  # pylint: disable=protected-access

    summary_args = parser.parse_args(["summary", str(tmp_path), "--strict"])
    export_args = parser.parse_args(
        ["export", str(tmp_path), str(tmp_path / "subject.npz"), "--strict"]
    )

    assert summary_args.strict is True
    assert export_args.strict is True
