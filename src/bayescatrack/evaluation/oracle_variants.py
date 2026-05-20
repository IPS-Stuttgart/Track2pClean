"""Oracle track variants for diagnosing Track2p benchmark ceilings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from bayescatrack.evaluation.complete_track_scores import (
    normalize_track_matrix,
    score_track_matrices,
)
from bayescatrack.matching import build_track_rows_from_matches
from bayescatrack.reference import Track2pReference


@dataclass(frozen=True)
class OracleVariantResult:
    """One oracle variant and its score dictionary."""

    variant: str
    seed_session: int
    max_gap: int | None
    track_matrix: np.ndarray
    scores: dict[str, float | int]

    def to_row(self) -> dict[str, float | int | str]:
        return {
            "variant": self.variant,
            "seed_session": int(self.seed_session),
            "max_gap": "none" if self.max_gap is None else int(self.max_gap),
            "oracle_tracks": int(self.track_matrix.shape[0]),
            **self.scores,
        }


def oracle_reference_rows(
    reference: Track2pReference,
    *,
    curated_only: bool = False,
) -> np.ndarray:
    """Return the direct reference matrix as the scoring upper bound."""

    return normalize_track_matrix(reference.filtered_indices(curated_only=curated_only))


def oracle_consecutive_link_tracks(
    reference: Track2pReference,
    *,
    curated_only: bool = False,
    seed_session: int = 0,
) -> np.ndarray:
    """Stitch oracle tracks from consecutive manual-GT pairwise links."""

    return oracle_gap_limited_link_tracks(
        reference,
        curated_only=curated_only,
        seed_session=seed_session,
        max_gap=1,
    )


def oracle_gap_limited_link_tracks(
    reference: Track2pReference,
    *,
    curated_only: bool = False,
    seed_session: int = 0,
    max_gap: int = 1,
) -> np.ndarray:
    """Build oracle rows using only GT links with a bounded session gap.

    This tests whether the allowed assignment graph is capable of reconstructing
    the reference when edge costs are perfect.  Tracks are seeded from
    ``seed_session`` and propagation stops across gaps larger than ``max_gap``.
    """

    seed_session = int(seed_session)
    if seed_session < 0 or seed_session >= reference.n_sessions:
        raise IndexError(
            f"seed_session {seed_session} out of bounds for {reference.n_sessions} sessions"
        )
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")

    reference_matrix = reference.filtered_indices(curated_only=curated_only)
    seed_rows = [
        row_index
        for row_index, row in enumerate(reference_matrix)
        if _is_present(row[seed_session])
    ]
    result = np.empty((len(seed_rows), reference.n_sessions), dtype=object)
    result[:] = None
    for output_index, reference_row_index in enumerate(seed_rows):
        row = reference_matrix[reference_row_index]
        result[output_index, seed_session] = int(row[seed_session])
        _propagate_oracle_row(result[output_index], row, seed_session, max_gap=max_gap)
    return normalize_track_matrix(result)


def oracle_pairwise_stitch_tracks(
    reference: Track2pReference,
    *,
    curated_only: bool = False,
    seed_session: int = 0,
) -> np.ndarray:
    """Use BayesCaTrack's normal row-stitcher on consecutive GT links."""

    reference_matrix = reference.filtered_indices(curated_only=curated_only)
    seed_rois = sorted(
        {
            int(row[seed_session])
            for row in reference_matrix
            if _is_present(row[seed_session])
        }
    )
    matches = [
        reference.pairwise_matches(
            session_index,
            session_index + 1,
            curated_only=curated_only,
        )
        for session_index in range(reference.n_sessions - 1)
    ]
    return build_track_rows_from_matches(
        reference.session_names,
        matches,
        start_roi_indices=seed_rois,
        start_session_index=seed_session,
        fill_value=-1,
    )


def score_oracle_variants(
    reference: Track2pReference,
    *,
    curated_only: bool = False,
    seed_sessions: Iterable[int] = (0,),
    max_gaps: Iterable[int] = (1, 2, 3),
) -> tuple[OracleVariantResult, ...]:
    """Score direct, stitched and gap-limited oracle variants."""

    reference_matrix = oracle_reference_rows(reference, curated_only=curated_only)
    results: list[OracleVariantResult] = []
    for seed_session in seed_sessions:
        seed_session = int(seed_session)
        direct = reference_matrix
        results.append(
            OracleVariantResult(
                variant="oracle_reference_rows",
                seed_session=seed_session,
                max_gap=None,
                track_matrix=direct,
                scores=score_track_matrices(direct, reference_matrix),
            )
        )
        stitched = oracle_pairwise_stitch_tracks(
            reference,
            curated_only=curated_only,
            seed_session=seed_session,
        )
        results.append(
            OracleVariantResult(
                variant="oracle_pairwise_stitch_consecutive",
                seed_session=seed_session,
                max_gap=1,
                track_matrix=stitched,
                scores=score_track_matrices(stitched, reference_matrix),
            )
        )
        for max_gap in max_gaps:
            max_gap = int(max_gap)
            gap_limited = oracle_gap_limited_link_tracks(
                reference,
                curated_only=curated_only,
                seed_session=seed_session,
                max_gap=max_gap,
            )
            results.append(
                OracleVariantResult(
                    variant="oracle_gap_limited_reference_links",
                    seed_session=seed_session,
                    max_gap=max_gap,
                    track_matrix=gap_limited,
                    scores=score_track_matrices(gap_limited, reference_matrix),
                )
            )
    return tuple(results)


def _propagate_oracle_row(
    output_row: np.ndarray,
    reference_row: Sequence[Any],
    seed_session: int,
    *,
    max_gap: int,
) -> None:
    last_present = seed_session
    for session_index in range(seed_session + 1, len(reference_row)):
        if not _is_present(reference_row[session_index]):
            continue
        if session_index - last_present > max_gap:
            break
        output_row[session_index] = int(reference_row[session_index])
        last_present = session_index

    last_present = seed_session
    for session_index in range(seed_session - 1, -1, -1):
        if not _is_present(reference_row[session_index]):
            continue
        if last_present - session_index > max_gap:
            break
        output_row[session_index] = int(reference_row[session_index])
        last_present = session_index


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


__all__ = [
    "OracleVariantResult",
    "oracle_consecutive_link_tracks",
    "oracle_gap_limited_link_tracks",
    "oracle_pairwise_stitch_tracks",
    "oracle_reference_rows",
    "score_oracle_variants",
]
