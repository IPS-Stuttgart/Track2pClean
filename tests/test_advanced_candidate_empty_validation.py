from __future__ import annotations

import numpy as np
from bayescatrack import advanced_roi_components
from bayescatrack._empty_candidate_margin import install_empty_candidate_gate_margin_fix
from bayescatrack.advanced_roi_components import candidate_mask_from_cost_matrix


def test_candidate_pruning_accepts_empty_rows_with_margin_gate():
    mask = candidate_mask_from_cost_matrix(
        np.zeros((0, 3), dtype=float),
        top_k=None,
        gate_margin=1.0,
    )

    assert mask.shape == (0, 3)
    assert mask.dtype == bool


def test_candidate_pruning_accepts_empty_columns_with_margin_gate():
    mask = candidate_mask_from_cost_matrix(
        np.zeros((3, 0), dtype=float),
        top_k=1,
        gate_margin=1.0,
    )

    assert mask.shape == (3, 0)
    assert mask.dtype == bool


def test_candidate_pruning_guard_reinstalls_after_stale_installed_flag(monkeypatch):
    def stale_candidate_mask(
        cost_matrix,
        *,
        top_k,
        include_columns=True,
        gate_margin=None,
        large_cost=1.0e6,
    ):
        del top_k, include_columns
        costs = np.asarray(cost_matrix, dtype=float)
        if costs.ndim != 2:
            raise ValueError("cost_matrix must be two-dimensional")
        admitted = np.isfinite(costs) & (costs < large_cost)
        if gate_margin is not None:
            safe = np.where(np.isfinite(costs), costs, large_cost)
            row_best = np.min(safe, axis=1, keepdims=True)
            col_best = np.min(safe, axis=0, keepdims=True)
            admitted &= (safe <= row_best + gate_margin) | (
                safe <= col_best + gate_margin
            )
        return admitted

    monkeypatch.setattr(
        advanced_roi_components,
        "candidate_mask_from_cost_matrix",
        stale_candidate_mask,
    )
    monkeypatch.setattr(
        advanced_roi_components,
        "_bayescatrack_empty_candidate_margin_fix",
        True,
        raising=False,
    )

    install_empty_candidate_gate_margin_fix()

    assert (
        advanced_roi_components.candidate_mask_from_cost_matrix
        is not stale_candidate_mask
    )
    mask = advanced_roi_components.candidate_mask_from_cost_matrix(
        np.zeros((3, 0), dtype=float),
        top_k=None,
        gate_margin=1.0,
    )
    assert mask.shape == (3, 0)
    assert mask.dtype == bool
