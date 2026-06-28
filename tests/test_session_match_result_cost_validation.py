import numpy as np
import pytest
from bayescatrack.matching import SessionMatchResult


def _result(costs):
    return SessionMatchResult(
        reference_session_name="s1",
        measurement_session_name="s2",
        reference_positions=[0],
        measurement_positions=[0],
        reference_roi_indices=[10],
        measurement_roi_indices=[100],
        costs=costs,
    )


@pytest.mark.parametrize(
    "costs", [[True], [np.bool_(False)], np.array([True], dtype=bool)]
)
def test_rejects_boolean_costs(costs):
    with pytest.raises(ValueError, match="finite numeric assignment costs"):
        _result(costs)


def test_accepts_numeric_cost_strings():
    assert _result(["1.25"]).costs.tolist() == [1.25]
