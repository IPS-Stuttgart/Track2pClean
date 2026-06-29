from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import load_track2p_subject


@pytest.mark.parametrize(
    "bad_input_format",
    [
        ["auto"],
        {"format": "auto"},
        np.asarray(["auto"], dtype=object),
    ],
)
def test_subject_loader_rejects_unhashable_input_format_values(
    tmp_path,
    bad_input_format,
):
    with pytest.raises(ValueError, match="input_format must be 'auto', 'suite2p', or 'npy'"):
        load_track2p_subject(tmp_path, input_format=bad_input_format)
