from __future__ import annotations

from typing import Any

import pytest
from bayescatrack.association.dynamic_edge_priors import (
    dynamic_edge_prior_config_from_mapping,
)


def test_empty_tuple_config_is_rejected() -> None:
    value: Any = tuple()
    with pytest.raises(ValueError, match="DynamicEdgePriorConfig"):
        dynamic_edge_prior_config_from_mapping(value)
