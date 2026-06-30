from __future__ import annotations

import pytest
from bayescatrack.experiments.benchmark_comparison import (
    build_subject_deficit_summary_rows,
    build_subject_gap_summary_rows,
)


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "1"])
def test_subject_gap_summary_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        build_subject_gap_summary_rows([], reference_approach=None, limit=limit)


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "1"])
def test_subject_deficit_summary_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        build_subject_deficit_summary_rows([], reference_approach=None, limit=limit)
