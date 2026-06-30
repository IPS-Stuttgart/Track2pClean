from __future__ import annotations

import pytest
from bayescatrack.experiments.track2p_policy_growth_veto_cleanup import GrowthVetoGate

_STRUCTURAL_FLAG_NAMES = (
    "require_not_suffix_edge",
    "require_terminal_edge",
    "require_last_session_edge",
    "require_complete_component",
)


@pytest.mark.parametrize("field_name", _STRUCTURAL_FLAG_NAMES)
@pytest.mark.parametrize("bad_value", ["false", "true", 1, 0])
def test_growth_veto_gate_rejects_non_boolean_structural_flags(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        GrowthVetoGate(**{field_name: bad_value})


def test_growth_veto_gate_accepts_explicit_boolean_structural_flags() -> None:
    gate = GrowthVetoGate(
        require_not_suffix_edge=False,
        require_terminal_edge=False,
        require_last_session_edge=False,
        require_complete_component=False,
    )

    assert gate.require_not_suffix_edge is False
    assert gate.require_terminal_edge is False
    assert gate.require_last_session_edge is False
    assert gate.require_complete_component is False
