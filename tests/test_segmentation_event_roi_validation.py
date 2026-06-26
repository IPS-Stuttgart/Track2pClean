from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.segmentation_events import (
    merge_event_candidates,
    split_event_candidates,
)


def _pairwise_components() -> dict[str, np.ndarray]:
    return {
        "overlap_min_fraction": np.asarray([[0.5]], dtype=float),
        "weighted_dice_similarity": np.asarray([[0.5]], dtype=float),
        "area_ratio_cost": np.asarray([[0.0]], dtype=float),
    }


@pytest.mark.parametrize("bad_indices", ([True], [np.bool_(False)], [1.5], [-1], [np.nan]))
def test_split_event_candidates_reject_invalid_reference_roi_indices(
    bad_indices: list[object],
) -> None:
    with pytest.raises(ValueError, match="reference_roi_indices"):
        split_event_candidates(
            _pairwise_components(),
            reference_roi_indices=bad_indices,  # type: ignore[arg-type]
            measurement_roi_indices=[0],
        )


@pytest.mark.parametrize("bad_indices", ([True], [np.bool_(False)], [1.5], [-1], [np.nan]))
def test_merge_event_candidates_reject_invalid_measurement_roi_indices(
    bad_indices: list[object],
) -> None:
    with pytest.raises(ValueError, match="measurement_roi_indices"):
        merge_event_candidates(
            _pairwise_components(),
            reference_roi_indices=[0],
            measurement_roi_indices=bad_indices,  # type: ignore[arg-type]
        )
