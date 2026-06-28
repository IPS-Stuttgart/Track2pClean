from __future__ import annotations

import pytest

from bayescatrack import load_track2p_subject


@pytest.mark.parametrize(
    "plane_name",
    [
        "",
        ".",
        "..",
        "../plane0",
        "nested/plane0",
        r"..\plane0",
        r"nested\plane0",
        "/tmp/plane0",
        r"C:\tmp\plane0",
        b"plane0",
    ],
)
def test_load_track2p_subject_rejects_path_like_plane_name(tmp_path, plane_name):
    subject_dir = tmp_path / "jm123"
    (subject_dir / "2024-05-01_a").mkdir(parents=True)

    with pytest.raises(ValueError, match="plane_name"):
        load_track2p_subject(subject_dir, plane_name=plane_name)
