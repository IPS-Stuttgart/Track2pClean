from __future__ import annotations

from typing import Any

import pytest
from bayescatrack.association.absence_model import absence_model_config_from_mapping
from bayescatrack.association.dynamic_edge_priors import (
    dynamic_edge_prior_config_from_mapping,
)
from bayescatrack.association.teacher_priors import (
    teacher_edge_prior_config_from_mapping,
)
from bayescatrack.association.track2p_policy_priors import (
    track2p_policy_prior_config_from_mapping,
)


def test_empty_tuple_config_is_rejected() -> None:
    value: Any = tuple()
    with pytest.raises(ValueError, match="DynamicEdgePriorConfig"):
        dynamic_edge_prior_config_from_mapping(value)


def test_absence_config_rejects_empty_tuple() -> None:
    value: Any = tuple()
    with pytest.raises(ValueError, match="AbsenceModelConfig"):
        absence_model_config_from_mapping(value)


def test_teacher_config_rejects_empty_tuple() -> None:
    value: Any = tuple()
    with pytest.raises(ValueError, match="TeacherEdgePriorConfig"):
        teacher_edge_prior_config_from_mapping(value)


def test_track2p_policy_config_rejects_empty_tuple() -> None:
    value: Any = tuple()
    with pytest.raises(ValueError, match="Track2pPolicyPriorConfig"):
        track2p_policy_prior_config_from_mapping(value)
