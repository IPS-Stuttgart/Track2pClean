"""Keep full-MHT sparse matrix source ROI metadata aligned with matrix rows.

``_sparse_pair_matrices`` filters requested source ROIs to the ROIs that are
actually present in the source session before materialising the feature matrices.
The returned ``source_indices`` metadata must be filtered in the same order;
otherwise downstream lookups can associate matrix row 0 with a stale source ROI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_full_mht_sparse_source_index_alignment_patch"
_CACHE_ATTRIBUTE = "_full_mht_sparse_matrices"


def install_full_mht_sparse_source_index_alignment() -> None:
    """Install an idempotent source-index alignment guard for full-MHT matrices."""

    try:
        from . import (  # pylint: disable=import-outside-toplevel
            track2p_policy_full_mht_benchmark as _full_mht,
        )
    except ImportError as exc:  # pragma: no cover - optional PyRecEst dependency
        if "pyrecest" in str(exc).lower() or "track2p-policy-full-mht requires" in str(exc):
            return
        raise

    original_sparse_pair_matrices = _full_mht._sparse_pair_matrices
    if getattr(original_sparse_pair_matrices, _PATCH_MARKER, False):
        return

    @wraps(original_sparse_pair_matrices)
    def _sparse_pair_matrices_with_aligned_sources(
        sessions: Sequence[Any],
        feature_cache: Any,
        *,
        source_session: int,
        target_session: int,
        source_rois: Sequence[int],
        edge_top_k: int,
        config: Any,
    ) -> Any:
        matrices = original_sparse_pair_matrices(
            sessions,
            feature_cache,
            source_session=int(source_session),
            target_session=int(target_session),
            source_rois=source_rois,
            edge_top_k=int(edge_top_k),
            config=config,
        )
        aligned_source_indices = _aligned_source_indices(
            _full_mht,
            sessions,
            source_session=int(source_session),
            source_rois=source_rois,
        )
        row_count = _matrix_row_count(matrices)
        if int(aligned_source_indices.shape[0]) != row_count:
            raise ValueError(
                "Full-MHT sparse source ROI metadata does not match feature matrix rows"
            )
        current_source_indices = np.asarray(matrices.source_indices, dtype=int).reshape(-1)
        if np.array_equal(current_source_indices, aligned_source_indices):
            return matrices

        fixed_matrices = replace(matrices, source_indices=aligned_source_indices)
        _store_fixed_matrix_in_cache(
            feature_cache,
            fixed_matrices,
            source_session=int(source_session),
            target_session=int(target_session),
            source_rois=source_rois,
            edge_top_k=int(edge_top_k),
            config=config,
        )
        return fixed_matrices

    setattr(_sparse_pair_matrices_with_aligned_sources, _PATCH_MARKER, True)
    setattr(
        _sparse_pair_matrices_with_aligned_sources,
        "_bayescatrack_original",
        original_sparse_pair_matrices,
    )
    _full_mht._sparse_pair_matrices = _sparse_pair_matrices_with_aligned_sources


def _aligned_source_indices(
    full_mht_module: Any,
    sessions: Sequence[Any],
    *,
    source_session: int,
    source_rois: Sequence[int],
) -> np.ndarray:
    available = {
        int(roi)
        for roi in full_mht_module._roi_indices(sessions[int(source_session)])
    }
    return np.asarray(
        [int(roi) for roi in source_rois if int(roi) in available],
        dtype=int,
    )


def _matrix_row_count(matrices: Any) -> int:
    registered_iou = np.asarray(matrices.registered_iou)
    if registered_iou.ndim != 2:
        raise ValueError("Full-MHT sparse registered-IoU matrix must be 2-D")
    return int(registered_iou.shape[0])


def _store_fixed_matrix_in_cache(
    feature_cache: Any,
    matrices: Any,
    *,
    source_session: int,
    target_session: int,
    source_rois: Sequence[int],
    edge_top_k: int,
    config: Any,
) -> None:
    sparse_cache = getattr(feature_cache, _CACHE_ATTRIBUTE, None)
    if not isinstance(sparse_cache, dict):
        return
    sparse_cache[
        _sparse_pair_matrices_cache_key(
            source_session=int(source_session),
            target_session=int(target_session),
            source_rois=source_rois,
            edge_top_k=int(edge_top_k),
            config=config,
        )
    ] = matrices


def _sparse_pair_matrices_cache_key(
    *,
    source_session: int,
    target_session: int,
    source_rois: Sequence[int],
    edge_top_k: int,
    config: Any,
) -> tuple[Any, ...]:
    return (
        int(source_session),
        int(target_session),
        tuple(int(roi) for roi in source_rois),
        int(edge_top_k),
        float(config.growth_anchor_min_registered_iou),
        float(config.growth_anchor_min_shifted_iou),
        float(config.growth_anchor_min_cell_probability),
    )


__all__ = ["install_full_mht_sparse_source_index_alignment"]
