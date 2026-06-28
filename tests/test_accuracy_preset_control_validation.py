from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.accuracy_presets import build_track2p_accuracy_presets


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_gap": True}, "max_gap must be a positive integer"),
        ({"max_gap": np.asarray([2])}, "max_gap must be a positive integer"),
        ({"max_gap": 0}, "max_gap must be a positive integer"),
        (
            {"cost_threshold": True},
            "cost_threshold must be a finite non-negative value or None",
        ),
        (
            {"cost_threshold": np.nan},
            "cost_threshold must be a finite non-negative value or None",
        ),
        (
            {"cost_threshold": np.asarray([6.0])},
            "cost_threshold must be a finite non-negative value or None",
        ),
        ({"progress": np.bool_(False)}, "progress must be a boolean"),
    ],
)
def test_accuracy_presets_reject_ambiguous_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_track2p_accuracy_presets("/data/track2p", **kwargs)


def test_accuracy_presets_normalize_shared_numeric_controls() -> None:
    presets = build_track2p_accuracy_presets(
        "/data/track2p",
        max_gap="3",
        cost_threshold=np.float64(5.5),
        progress=False,
    )

    pruned = presets[1]
    supported_gap = presets[4]
    confidence_gap = presets[5]

    assert presets[0].config.max_gap == 3
    assert presets[0].config.cost_threshold == pytest.approx(5.5)
    assert presets[0].config.progress is False

    assert pruned.config.candidate_pruning_config is not None
    assert pruned.config.candidate_pruning_config["max_cost"] == pytest.approx(5.5)
    assert pruned.config.track2p_policy_prior_config is not None
    assert pruned.config.track2p_policy_prior_config["max_gap"] == 3

    assert supported_gap.config.max_gap == 3
    assert confidence_gap.runner_kwargs is not None
    assert confidence_gap.runner_kwargs["max_gap"] == 3
