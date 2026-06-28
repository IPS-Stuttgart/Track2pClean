from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.activity_similarity import activity_similarity_components


def _plane(n_rois: int) -> SimpleNamespace:
    return SimpleNamespace(
        n_rois=n_rois,
        traces=np.zeros((n_rois, 3), dtype=float),
        spike_traces=None,
        neuropil_traces=None,
    )


@pytest.mark.parametrize(
    ("kwarg", "bad_value"),
    (
        ("similarity_epsilon", "1e-6"),
        ("similarity_epsilon", b"1e-6"),
        ("similarity_epsilon", bytearray(b"1e-6")),
        ("similarity_epsilon", np.str_("1e-6")),
        ("similarity_epsilon", np.bytes_(b"1e-6")),
        ("similarity_epsilon", np.asarray("1e-6")),
        ("similarity_epsilon", np.asarray(b"1e-6")),
        ("event_threshold", "0.0"),
        ("event_threshold", b"0.0"),
        ("event_threshold", bytearray(b"0.0")),
        ("event_threshold", np.str_("0.0")),
        ("event_threshold", np.bytes_(b"0.0")),
        ("event_threshold", np.asarray("0.0")),
        ("event_threshold", np.asarray(b"0.0")),
    ),
)
def test_activity_similarity_rejects_text_like_scalar_controls(
    kwarg: str,
    bad_value: object,
) -> None:
    reference = _plane(1)
    measurement = _plane(1)

    with pytest.raises(ValueError, match=kwarg):
        activity_similarity_components(reference, measurement, **{kwarg: bad_value})


def test_activity_similarity_accepts_numpy_numeric_scalar_controls() -> None:
    reference = _plane(1)
    measurement = _plane(1)

    components = activity_similarity_components(
        reference,
        measurement,
        similarity_epsilon=np.asarray(1.0e-6, dtype=float),
        event_threshold=np.asarray(0.0, dtype=float),
    )

    assert components["activity_similarity_cost"].shape == (1, 1)
