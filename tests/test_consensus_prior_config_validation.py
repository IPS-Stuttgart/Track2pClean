from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.consensus_priors import (
    apply_consensus_edge_priors,
    consensus_prior_config_from_mapping,
)


@pytest.mark.parametrize("bad_config", ["", [], ()])
def test_consensus_prior_config_rejects_non_mapping_inputs(bad_config: object) -> None:
    with pytest.raises(ValueError, match="ConsensusPriorConfig"):
        consensus_prior_config_from_mapping(bad_config)  # type: ignore[arg-type]


def test_apply_consensus_edge_priors_rejects_empty_string_config() -> None:
    with pytest.raises(ValueError, match="ConsensusPriorConfig"):
        apply_consensus_edge_priors(
            {(0, 1): np.array([[1.0]])},
            {},
            config="",  # type: ignore[arg-type]
        )


def test_consensus_prior_config_accepts_mapping() -> None:
    config = consensus_prior_config_from_mapping({"relief": 0.125})

    assert config is not None
    assert config.relief == 0.125
