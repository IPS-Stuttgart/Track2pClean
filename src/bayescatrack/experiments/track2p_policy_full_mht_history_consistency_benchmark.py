"""FullMHT benchmark variant with identity-history consistency scoring.

This runner is intentionally a thin experimental wrapper around
``track2p-policy-full-mht``.  The base FullMHT row scores candidate continuations
mostly from the current scan pair.  This variant keeps the same assignment beam
and PyRecEst/Murty machinery, but subtracts a label-free per-track history risk
before scan assignment.  A candidate is penalized only when it is jointly weak in
overlap/cell evidence and anomalous in growth/motion evidence relative to that
same track's previous accepted edges.

The implementation is default-off and does not read manual-GT columns.  It is a
method-development row for testing whether full-history likelihoods can make MHT
responsible for identity selection instead of merely replaying local edge scores.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from bayescatrack.experiments import track2p_policy_full_mht_benchmark as base
from bayescatrack.experiments.full_mht_history_consistency_model import (
    IdentityHistoryConsistencyConfig,
    identity_history_consistency_risk,
)

METHOD = "track2p-policy-full-mht-history-consistency"

_HISTORY_CONFIG = IdentityHistoryConsistencyConfig()


def _history_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--history-consistency-weight", type=float, default=0.0)
    parser.add_argument("--history-consistency-min-history-edges", type=int, default=2)
    parser.add_argument("--history-consistency-min-feature-scale", type=float, default=0.05)
    parser.add_argument("--history-consistency-joint-margin", type=float, default=1.0)
    parser.add_argument("--history-consistency-score-clip", type=float, default=8.0)
    return parser


def _history_config_from_args(
    argv: Sequence[str],
) -> tuple[IdentityHistoryConsistencyConfig, list[str]]:
    namespace, remaining = _history_arg_parser().parse_known_args(list(argv))
    return (
        IdentityHistoryConsistencyConfig(
            weight=float(namespace.history_consistency_weight),
            min_history_edges=max(1, int(namespace.history_consistency_min_history_edges)),
            min_feature_scale=float(namespace.history_consistency_min_feature_scale),
            joint_margin=float(namespace.history_consistency_joint_margin),
            score_clip=float(namespace.history_consistency_score_clip),
        ),
        list(remaining),
    )


@contextmanager
def _patched_full_mht_runner(history_config: IdentityHistoryConsistencyConfig):
    global _HISTORY_CONFIG
    previous_config = _HISTORY_CONFIG
    previous_expand = base._expand_hypothesis_scan
    previous_method = base.METHOD
    _HISTORY_CONFIG = history_config
    base._expand_hypothesis_scan = _expand_hypothesis_scan_with_history_consistency
    base.METHOD = METHOD
    try:
        yield
    finally:
        base.METHOD = previous_method
        base._expand_hypothesis_scan = previous_expand
        _HISTORY_CONFIG = previous_config


def main(argv: list[str] | None = None) -> int:
    history_config, base_args = _history_config_from_args([] if argv is None else argv)
    with _patched_full_mht_runner(history_config):
        return int(base.main(base_args))


def _expand_hypothesis_scan_with_history_consistency(
    hypothesis: Any,
    *,
    sessions: Sequence[Any],
    feature_cache: Any,
    session_index: int,
    config: Any,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]] | None = None,
    scan_pair_matrices_by_source_session: Mapping[int, Any] | None = None,
) -> list[Any]:
    tracks = np.asarray(hypothesis.tracks, dtype=int)
    next_session = int(session_index) + 1
    prior_edges = track2p_prior_edges or frozenset()
    active_sources = base._active_track_sources(
        tracks, session_index=int(session_index), max_gap=int(config.max_gap)
    )
    if not active_sources:
        carried = tracks.copy()
        return [
            base._MHTHypothesis(
                carried,
                hypothesis.score,
                hypothesis.history
                + (
                    {
                        "session_index": int(session_index),
                        "scan_cost": 0.0,
                        "assigned_edges": 0,
                        "missed_tracks": 0,
                        "missed_prior_successors": 0,
                        "switched_prior_successors": 0,
                        "no_prior_successor_continuations": 0,
                        "selected_prior_risk": 0.0,
                        "selected_history_consistency_risk": 0.0,
                        "gap_active_tracks": 0,
                        "gap_reactivated_tracks": 0,
                        "max_gap_length": 0,
                        "scan_candidates": 0,
                        "growth_anchor_count": 0,
                        "growth_model_type": "no_active_tracks",
                    },
                ),
            )
        ]

    source_rois_by_session: dict[int, list[int]] = {}
    for active_source in active_sources:
        source_rois_by_session.setdefault(int(active_source.source_session), []).append(
            int(active_source.source_roi)
        )
    matrices_by_source_session = {
        int(source_session): base._pair_matrices_for_active_sources(
            sessions,
            feature_cache,
            source_session=int(source_session),
            target_session=next_session,
            source_rois=source_rois,
            edge_top_k=int(config.edge_top_k),
            config=config,
            track2p_prior_edges=prior_edges,
            scan_pair_matrices_by_source_session=scan_pair_matrices_by_source_session,
        )
        for source_session, source_rois in source_rois_by_session.items()
    }
    matrix_diagnostics = base._matrix_diagnostics(matrices_by_source_session)
    finite_target_rois = sorted(
        {
            int(target_roi)
            for matrices in matrices_by_source_session.values()
            for target_roi in np.asarray(matrices.target_indices, dtype=int)
            if base._cell_probability(sessions, next_session, int(target_roi))
            >= float(feature_cache.cell_probability_threshold)
        }
    )
    active_rows = [int(active_source.row_index) for active_source in active_sources]
    gap_active_tracks = sum(1 for active in active_sources if int(active.gap_length) > 0)
    max_gap_length = max((int(active.gap_length) for active in active_sources), default=0)
    row_non_assignment_costs = np.asarray(
        [
            base._miss_cost(
                active_source,
                target_session=next_session,
                track2p_prior_edges=prior_edges,
                config=config,
            )
            for active_source in active_sources
        ],
        dtype=float,
    )
    all_miss_cost = float(np.sum(row_non_assignment_costs))
    all_missed_prior_successors = sum(
        1
        for active_source in active_sources
        if base._has_prior_successor(
            active_source,
            target_session=next_session,
            track2p_prior_edges=prior_edges,
        )
    )
    if not finite_target_rois:
        return _all_miss_hypothesis(
            hypothesis,
            tracks,
            active_rows=active_rows,
            next_session=next_session,
            session_index=session_index,
            active_count=len(active_sources),
            scan_cost=all_miss_cost,
            missed_prior_successors=all_missed_prior_successors,
            gap_active_tracks=gap_active_tracks,
            max_gap_length=max_gap_length,
            candidate_count=0,
            matrix_diagnostics=matrix_diagnostics,
        )

    source_lookup_by_session = {
        int(source_session): {
            int(roi): idx for idx, roi in enumerate(matrices.source_indices)
        }
        for source_session, matrices in matrices_by_source_session.items()
    }
    target_lookup_by_session = {
        int(source_session): {
            int(roi): idx for idx, roi in enumerate(matrices.target_indices)
        }
        for source_session, matrices in matrices_by_source_session.items()
    }
    candidate_entries: list[tuple[int, int, float, float]] = []
    candidate_target_roi_set: set[int] = set()
    candidate_count = 0
    for row_pos, active_source in enumerate(active_sources):
        source_session = int(active_source.source_session)
        matrices = matrices_by_source_session[source_session]
        source_lookup = source_lookup_by_session[source_session]
        target_lookup = target_lookup_by_session[source_session]
        source_local = source_lookup.get(int(active_source.source_roi))
        if source_local is None:
            continue
        row_scores: list[tuple[float, int, float]] = []
        for target_roi in finite_target_rois:
            target_local = target_lookup.get(int(target_roi))
            if target_local is None:
                continue
            score = base._edge_score(
                sessions,
                matrices,
                target_session=next_session,
                source_local=int(source_local),
                target_local=int(target_local),
                config=config,
                track2p_prior_edges=prior_edges,
            )
            history_risk = _candidate_history_consistency_risk(
                sessions,
                feature_cache,
                tracks,
                matrices,
                active_source=active_source,
                source_local=int(source_local),
                target_local=int(target_local),
                target_session=next_session,
                config=config,
                track2p_prior_edges=prior_edges,
            )
            score -= float(history_risk)
            if int(active_source.gap_length) > 0:
                score -= float(config.gap_reactivation_cost) * float(
                    active_source.gap_length
                )
            if score >= float(config.min_edge_score):
                row_scores.append((float(score), int(target_roi), float(history_risk)))
        row_scores.sort(reverse=True)
        for score, target_roi, history_risk in row_scores[
            : max(1, int(config.edge_top_k))
        ]:
            candidate_entries.append(
                (int(row_pos), int(target_roi), float(score), float(history_risk))
            )
            candidate_target_roi_set.add(int(target_roi))
            candidate_count += 1

    candidate_target_rois = sorted(candidate_target_roi_set)
    candidate_col_by_roi = {
        int(target_roi): idx for idx, target_roi in enumerate(candidate_target_rois)
    }
    candidate_history_risk_by_pair: dict[tuple[int, int], float] = {}
    cost_matrix = np.full(
        (len(active_sources), len(candidate_target_rois)), np.inf, dtype=float
    )
    for row_pos, target_roi, score, history_risk in candidate_entries:
        compact_col = int(candidate_col_by_roi[int(target_roi)])
        cost_matrix[int(row_pos), compact_col] = -float(score)
        candidate_history_risk_by_pair[(int(row_pos), compact_col)] = float(history_risk)

    if not np.isfinite(cost_matrix).any():
        return _all_miss_hypothesis(
            hypothesis,
            tracks,
            active_rows=active_rows,
            next_session=next_session,
            session_index=session_index,
            active_count=len(active_sources),
            scan_cost=all_miss_cost,
            missed_prior_successors=all_missed_prior_successors,
            gap_active_tracks=gap_active_tracks,
            max_gap_length=max_gap_length,
            candidate_count=candidate_count,
            matrix_diagnostics=matrix_diagnostics,
        )

    solutions, assignment_diagnostics = base._scan_assignment_solutions(
        cost_matrix,
        k=max(1, int(config.scan_hypotheses)),
        row_non_assignment_costs=row_non_assignment_costs,
        col_non_assignment_costs=np.zeros((len(candidate_target_rois),), dtype=float),
    )
    output: list[Any] = []
    for solution in solutions:
        assignment = np.asarray(solution["assignment"], dtype=int)
        updated = tracks.copy()
        assigned_edges = 0
        missed_tracks = 0
        gap_reactivated_tracks = 0
        selected_edge_summaries: list[str] = []
        selected_prior_edges = 0
        selected_non_prior_edges = 0
        missed_prior_successors = 0
        switched_prior_successors = 0
        no_prior_successor_continuations = 0
        selected_prior_risk = 0.0
        selected_history_consistency_risk = 0.0
        for row_pos, active_source in enumerate(active_sources):
            compact_col = int(assignment[int(row_pos)])
            row_index = int(active_source.row_index)
            if compact_col >= 0:
                target_roi = int(candidate_target_rois[compact_col])
                updated[row_index, next_session] = target_roi
                assigned_edges += 1
                selected_history_consistency_risk += float(
                    candidate_history_risk_by_pair.get((int(row_pos), compact_col), 0.0)
                )
                edge_summary = base._selected_edge_summary(
                    sessions,
                    matrices_by_source_session[int(active_source.source_session)],
                    active_source=active_source,
                    target_session=next_session,
                    target_roi=target_roi,
                    config=config,
                    track2p_prior_edges=prior_edges,
                )
                selected_edge_summaries.append(str(edge_summary["summary"]))
                if int(edge_summary["is_track2p_prior"]):
                    selected_prior_edges += 1
                    selected_prior_risk += float(
                        edge_summary.get("track2p_prior_risk", 0.0)
                    )
                else:
                    selected_non_prior_edges += 1
                    if base._has_prior_successor(
                        active_source,
                        target_session=next_session,
                        track2p_prior_edges=prior_edges,
                    ):
                        switched_prior_successors += 1
                    else:
                        no_prior_successor_continuations += 1
                if int(active_source.gap_length) > 0:
                    gap_reactivated_tracks += 1
            else:
                updated[row_index, next_session] = -1
                missed_tracks += 1
                if base._has_prior_successor(
                    active_source,
                    target_session=next_session,
                    track2p_prior_edges=prior_edges,
                ):
                    missed_prior_successors += 1
        scan_cost = float(solution["cost"])
        output.append(
            base._MHTHypothesis(
                updated,
                float(hypothesis.score) - scan_cost,
                hypothesis.history
                + (
                    {
                        "session_index": int(session_index),
                        "scan_cost": scan_cost,
                        "assigned_edges": int(assigned_edges),
                        "missed_tracks": int(missed_tracks),
                        "selected_prior_edges": int(selected_prior_edges),
                        "selected_non_prior_edges": int(selected_non_prior_edges),
                        "missed_prior_successors": int(missed_prior_successors),
                        "switched_prior_successors": int(switched_prior_successors),
                        "no_prior_successor_continuations": int(
                            no_prior_successor_continuations
                        ),
                        "selected_prior_risk": float(selected_prior_risk),
                        "selected_history_consistency_risk": float(
                            selected_history_consistency_risk
                        ),
                        "selected_edge_summaries": ";".join(selected_edge_summaries),
                        "gap_active_tracks": int(gap_active_tracks),
                        "gap_reactivated_tracks": int(gap_reactivated_tracks),
                        "max_gap_length": int(max_gap_length),
                        "scan_candidates": int(candidate_count),
                        **matrix_diagnostics,
                        **assignment_diagnostics,
                    },
                ),
            )
        )
    return output


def _all_miss_hypothesis(
    hypothesis: Any,
    tracks: np.ndarray,
    *,
    active_rows: Sequence[int],
    next_session: int,
    session_index: int,
    active_count: int,
    scan_cost: float,
    missed_prior_successors: int,
    gap_active_tracks: int,
    max_gap_length: int,
    candidate_count: int,
    matrix_diagnostics: Mapping[str, Any],
) -> list[Any]:
    assignment_diagnostics = base._empty_assignment_diagnostics(int(active_count))
    carried = np.asarray(tracks, dtype=int).copy()
    carried[np.asarray(active_rows, dtype=int), int(next_session)] = -1
    return [
        base._MHTHypothesis(
            carried,
            float(hypothesis.score) - float(scan_cost),
            hypothesis.history
            + (
                {
                    "session_index": int(session_index),
                    "scan_cost": float(scan_cost),
                    "assigned_edges": 0,
                    "missed_tracks": int(active_count),
                    "missed_prior_successors": int(missed_prior_successors),
                    "switched_prior_successors": 0,
                    "no_prior_successor_continuations": 0,
                    "selected_prior_risk": 0.0,
                    "selected_history_consistency_risk": 0.0,
                    "gap_active_tracks": int(gap_active_tracks),
                    "gap_reactivated_tracks": 0,
                    "max_gap_length": int(max_gap_length),
                    "scan_candidates": int(candidate_count),
                    **matrix_diagnostics,
                    **assignment_diagnostics,
                },
            ),
        )
    ]


def _candidate_history_consistency_risk(
    sessions: Sequence[Any],
    feature_cache: Any,
    tracks: np.ndarray,
    matrices: Any,
    *,
    active_source: Any,
    source_local: int,
    target_local: int,
    target_session: int,
    config: Any,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
) -> float:
    if float(_HISTORY_CONFIG.weight) <= 0.0:
        return 0.0
    row = np.asarray(tracks[int(active_source.row_index)], dtype=int)
    history_edges = _track_history_feature_rows(
        sessions,
        feature_cache,
        row,
        stop_session=int(active_source.source_session),
        config=config,
        track2p_prior_edges=track2p_prior_edges,
    )
    candidate = _edge_feature_row(
        sessions,
        matrices,
        source_local=int(source_local),
        target_local=int(target_local),
        target_session=int(target_session),
    )
    return identity_history_consistency_risk(
        history_edges,
        candidate,
        config=_HISTORY_CONFIG,
    )


def _track_history_feature_rows(
    sessions: Sequence[Any],
    feature_cache: Any,
    row: np.ndarray,
    *,
    stop_session: int,
    config: Any,
    track2p_prior_edges: frozenset[tuple[int, int, int, int]],
) -> tuple[dict[str, float], ...]:
    rows: list[dict[str, float]] = []
    stop = min(int(stop_session), int(row.shape[0]) - 1)
    for session_a in range(max(0, stop)):
        session_b = int(session_a) + 1
        roi_a = int(row[int(session_a)])
        roi_b = int(row[int(session_b)])
        if roi_a < 0 or roi_b < 0:
            continue
        matrices = base._sparse_pair_matrices(
            sessions,
            feature_cache,
            source_session=int(session_a),
            target_session=int(session_b),
            source_rois=(roi_a,),
            edge_top_k=max(1, int(config.edge_top_k)),
            config=config,
            track2p_prior_edges=track2p_prior_edges,
        )
        source_matches = np.flatnonzero(np.asarray(matrices.source_indices, dtype=int) == roi_a)
        target_matches = np.flatnonzero(np.asarray(matrices.target_indices, dtype=int) == roi_b)
        if source_matches.size == 0 or target_matches.size == 0:
            continue
        rows.append(
            _edge_feature_row(
                sessions,
                matrices,
                source_local=int(source_matches[0]),
                target_local=int(target_matches[0]),
                target_session=int(session_b),
            )
        )
    return tuple(rows)


def _edge_feature_row(
    sessions: Sequence[Any],
    matrices: Any,
    *,
    source_local: int,
    target_local: int,
    target_session: int,
) -> dict[str, float]:
    source_roi = int(matrices.source_indices[int(source_local)])
    target_roi = int(matrices.target_indices[int(target_local)])
    cell_a = base._cell_probability(sessions, int(matrices.source_session), source_roi)
    cell_b = base._cell_probability(sessions, int(target_session), target_roi)
    return {
        "registered_iou": base._finite_float(
            matrices.registered_iou[int(source_local), int(target_local)], 0.0
        ),
        "shifted_iou": base._finite_float(
            matrices.shifted_iou[int(source_local), int(target_local)], 0.0
        ),
        "min_cell_probability": min(float(cell_a), float(cell_b)),
        "growth_residual": base._finite_float(
            matrices.growth_residual[int(source_local), int(target_local)], 0.0
        ),
        "growth_mahalanobis": base._finite_float(
            matrices.growth_mahalanobis[int(source_local), int(target_local)], 0.0
        ),
        "local_deformation": base._finite_float(
            matrices.local_deformation[int(source_local), int(target_local)], 0.0
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
