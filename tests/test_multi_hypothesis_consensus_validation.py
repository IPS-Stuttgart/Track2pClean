from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.multi_hypothesis import consensus_edges


@pytest.mark.parametrize(
    ("track_matrices", "message"),
    [
        ((((0, 1, 0, True),),), "integer entries"),
        (([[0, 1.5]],), "integer entries"),
        (([[0, np.nan]],), "integer entries"),
        (([[0, -2]],), "non-negative ROI indices or fill_value"),
        ((((0, 1, 0, -1),),), "explicit edge sets must contain non-negative"),
    ],
)
def test_consensus_edges_rejects_malformed_consensus_entries(track_matrices, message):
    with pytest.raises(ValueError, match=message):
        consensus_edges(track_matrices)


def test_consensus_edges_rejects_boolean_fill_value():
    with pytest.raises(ValueError, match="fill_value"):
        consensus_edges(([[0, -1]],), fill_value=True)
