from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.track2p_policy_priors import (
    Track2pPolicyPriorConfig,
    apply_track2p_policy_edge_prior,
)


@pytest.mark.parametrize(
    "consecutive_only",
    ["false", "true", "", 0, 1, 0.0, 1.0, np.array(True), None],
)
def test_policy_prior_config_rejects_ambiguous_consecutive_only(
    consecutive_only: object,
) -> None:
    with pytest.raises(ValueError, match="consecutive_only"):
        Track2pPolicyPriorConfig(
            consecutive_only=consecutive_only,  # type: ignore[arg-type]
        )


def test_policy_prior_config_accepts_numpy_boolean_consecutive_only() -> None:
    config = Track2pPolicyPriorConfig(consecutive_only=np.bool_(True))

    assert config.consecutive_only is True


def test_policy_prior_mapping_rejects_string_false_consecutive_only() -> None:
    costs = np.asarray([[5.0]], dtype=float)
    components = {"iou": np.asarray([[0.95]], dtype=float)}

    with pytest.raises(ValueError, match="consecutive_only"):
        apply_track2p_policy_edge_prior(
            costs,
            components,
            session_gap=2,
            config={"consecutive_only": "false", "relief": 1.0},
        )
