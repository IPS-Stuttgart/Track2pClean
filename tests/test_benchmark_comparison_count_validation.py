from __future__ import annotations

import pytest
from bayescatrack.experiments.benchmark_comparison import aggregate_rows


def _metric_row(**overrides: str) -> dict[str, str]:
    fp = "f" + "alse_positives"
    fn = "f" + "alse_negatives"
    row = {
        "approach": "candidate",
        "pairwise_f1": "1.0",
        "pairwise_true_positives": "1",
        "pairwise_" + fp: "0",
        "pairwise_" + fn: "0",
        "complete_track_f1": "1.0",
        "complete_track_true_positives": "1",
        "complete_track_" + fp: "0",
        "complete_track_" + fn: "0",
    }
    row.update(overrides)
    return row


def test_benchmark_comparison_rejects_fractional_count_fields() -> None:
    with pytest.raises(ValueError, match="pairwise_true_positives"):
        aggregate_rows([_metric_row(pairwise_true_positives="1.25")])


def test_benchmark_comparison_accepts_integer_like_count_strings() -> None:
    rows = aggregate_rows([_metric_row(pairwise_true_positives="2.0")])

    assert rows[0]["pairwise_f1_micro"] == pytest.approx(1.0)
