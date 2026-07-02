import numpy as np
import pytest

from bayescatrack.reference import Track2pReference, score_complete_tracks_against_reference


def test_reference_subset_ct_uses_same_seed_restricted_universe():
    reference = Track2pReference(
        session_names=("s0", "s1", "s2"),
        suite2p_indices=np.array(
            [
                [0, 10, 20],
                [None, 11, 21],
            ],
            dtype=object,
        ),
    )

    scores = score_complete_tracks_against_reference(
        np.array([[0, 10, 20]], dtype=object),
        reference,
        session_indices=(1, 2),
        seed_session=0,
        restrict_to_reference_seed_rois=True,
    )

    assert scores["T_rc"] == 1
    assert scores["T_c"] == 1
    assert scores["T_gt"] == 1
    assert scores["ct"] == pytest.approx(1.0)
