from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import export_subject_to_npz
from bayescatrack.core import _bridge_impl


@pytest.mark.parametrize(
    ("option_name", "bad_value"),
    [
        ("include_behavior", "false"),
        ("include_masks", "false"),
        ("weighted", 1),
        ("validate_pyrecest", np.asarray(True)),
    ],
)
def test_export_subject_to_npz_rejects_ambiguous_boolean_options(
    tmp_path, option_name, bad_value
):
    with pytest.raises(ValueError, match=f"{option_name} must be a boolean"):
        export_subject_to_npz(
            tmp_path / "subject",
            tmp_path / "export.npz",
            **{option_name: bad_value},
        )


def test_export_subject_to_npz_accepts_numpy_boolean_options(tmp_path, monkeypatch):
    def fake_load_track2p_subject(*args, **kwargs):
        assert kwargs["include_behavior"] is False
        return []

    monkeypatch.setattr(_bridge_impl, "load_track2p_subject", fake_load_track2p_subject)

    summary = export_subject_to_npz(
        tmp_path / "subject",
        tmp_path / "export.npz",
        include_behavior=np.bool_(False),
        include_masks=np.bool_(False),
        weighted=np.bool_(False),
        validate_pyrecest=np.bool_(False),
    )

    assert summary["n_sessions"] == 0
    assert (tmp_path / "export.npz").exists()
