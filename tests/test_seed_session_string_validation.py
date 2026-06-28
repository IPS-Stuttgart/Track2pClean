from __future__ import annotations

from pathlib import Path

import pytest
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _resolved_seed_sessions,
)


@pytest.mark.parametrize("seed_session", ["1", b"1"])
def test_seed_session_rejects_string_like_scalars(seed_session: object) -> None:
    config = Track2pBenchmarkConfig(
        data=Path("."),
        method="track2p-baseline",
        seed_session=seed_session,  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="seed_session must contain integer session indices",
    ):
        _resolved_seed_sessions(config, n_sessions=4)


@pytest.mark.parametrize("seed_sessions", [("1",), (b"1",)])
def test_seed_sessions_rejects_string_like_iterable_values(
    seed_sessions: tuple[object, ...],
) -> None:
    config = Track2pBenchmarkConfig(
        data=Path("."),
        method="track2p-baseline",
        seed_sessions=seed_sessions,  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="seed_sessions must contain integer session indices",
    ):
        _resolved_seed_sessions(config, n_sessions=4)
