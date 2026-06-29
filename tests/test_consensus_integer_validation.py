from __future__ import annotations

import pytest
from bayescatrack.association.multi_hypothesis import consensus_edges


class _BadIndex:
    def __init__(self, exc_type: type[BaseException]) -> None:
        self._exc_type = exc_type

    def __index__(self) -> int:
        raise self._exc_type("bad integer conversion")


@pytest.mark.parametrize("exc_type", [ValueError, OverflowError, ArithmeticError])
def test_consensus_edges_normalizes_index_protocol_errors(
    exc_type: type[BaseException],
) -> None:
    track_matrix = [[_BadIndex(exc_type), 1]]

    with pytest.raises(ValueError, match="integer entries"):
        consensus_edges((track_matrix,), min_votes=1)
