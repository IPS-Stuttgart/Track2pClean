from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.edge_ranking import summarize_edge_ranking_rows


class _ValueErrorIndex:
    def __index__(self) -> int:
        raise ValueError("adapter index failure")


class _OverflowIndex:
    def __index__(self) -> int:
        raise OverflowError("adapter index overflow")


def _edge_row() -> dict[str, float | int | str]:
    return {
        "subject": "jm_test",
        "session_a": 0,
        "session_b": 1,
        "session_gap": 1,
        "score_name": "cost",
        "edge_present": 1,
        "true_is_finite": 1,
        "row_rank": 1,
        "column_rank": 1,
        "row_margin": 0.5,
        "column_margin": 0.5,
    }


@pytest.mark.parametrize(
    "cutoffs",
    [
        (),
        (True,),
        (np.bool_(False),),
        (0,),
        (-1,),
        (1.5,),
        (np.inf,),
        ("1",),
        (_ValueErrorIndex(),),
        (_OverflowIndex(),),
        (1, 1),
    ],
)
def test_summarize_edge_ranking_rows_rejects_malformed_cutoffs(cutoffs) -> None:
    with pytest.raises(ValueError, match="hit_ks"):
        summarize_edge_ranking_rows([_edge_row()], hit_ks=cutoffs)


def test_summarize_edge_ranking_rows_accepts_integer_like_cutoffs() -> None:
    summary = summarize_edge_ranking_rows(
        [_edge_row()],
        hit_ks=(1, np.int64(3), 5.0),
    )

    assert summary[0]["row_hit_at_1"] == pytest.approx(1.0)
    assert summary[0]["row_hit_at_3"] == pytest.approx(1.0)
    assert summary[0]["row_hit_at_5"] == pytest.approx(1.0)
