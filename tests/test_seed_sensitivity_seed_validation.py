from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.experiments import track2p_policy_seed_sensitivity_audit as audit


@pytest.mark.parametrize(
    "seed_sessions",
    [
        (True,),
        (np.bool_(False),),
        (1.5,),
        (np.float64(2.5),),
        (np.nan,),
        (np.inf,),
        "1.5",
        "0,,2",
    ],
)
def test_seed_sensitivity_seed_resolver_rejects_silent_index_coercions(
    seed_sessions,
) -> None:
    with pytest.raises(ValueError, match="seed_sessions"):
        audit._resolved_seed_sessions(seed_sessions, n_sessions=3)


@pytest.mark.parametrize("n_sessions", [True, 1.5, np.nan, 0, -1])
def test_seed_sensitivity_seed_resolver_rejects_invalid_session_count(
    n_sessions,
) -> None:
    with pytest.raises(ValueError, match="n_sessions"):
        audit._resolved_seed_sessions("all", n_sessions=n_sessions)


def test_seed_sensitivity_seed_resolver_accepts_numpy_integer_sequence() -> None:
    assert audit._resolved_seed_sessions(
        np.asarray([0, 2], dtype=np.int64), n_sessions=3
    ) == (0, 2)
