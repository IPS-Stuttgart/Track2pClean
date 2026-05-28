"""Support-gated gap rescue followed by component cleanup for Track2p-policy rows.

The raw gap-rescue policy can admit many direct skip links.  Those links are
useful when they repair an isolated missing adjacent continuation, but they are
risky when they are completely isolated from the adjacent-session graph.  This
runner keeps a direct skip link only when the skipped interval has adjacent
support at either endpoint, then uses the existing lookahead gap propagation and
weakest-bridge component cleanup.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_emulation_benchmark import (
    _roi_indices,
    _thresholded_links_by_gap,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentAuditOutput,
    ComponentCleanupConfig,
    _evaluated_prediction_rows,
    _mark_applied_splits,
    _normalize_int_track_matrix,
    apply_weakest_bridge_splits,
    component_audit_rows,
    write_component_rows,
)
from bayescatrack.experiments.track2p_policy_gap_component_cleanup import (
    TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MIN_SIDE_OBSERVATIONS,
    TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_REQUIRE_COMPLETE_TRACK,
    _default_gap_cleanup_config,
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_gap_pruned import tracks_from_gap_links
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_SUPPORTED_GAP_COMPONENT_CLEANUP_METHOD = (
    "track2p-policy-supported-gap-component-cleanup"
)
TRACK2P_POLICY_SUPPORTED_GAP_DEFAULT_MIN_BRIDGE_SUPPORT = 1


def emulate_track2p_supported_gap_tracks(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str = TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    max_gap: int = TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP,
    min_bridge_support: int = TRACK2P_POLICY_SUPPORTED_GAP_DEFAULT_MIN_BRIDGE_SUPPORT,
) -> np.ndarray:
    """Return tracks from direct gap links with adjacent endpoint support.

    For a skip edge ``t -> t + k`` with ``k > 1``, bridge support is counted from
    accepted adjacent links touching the skipped interval: one vote when the
    source ROI has an accepted outgoing adjacent link into the interval and one
    vote when the target ROI has an accepted incoming adjacent link from the
    interval.  Requiring one vote removes fully isolated direct skip matches;
    requiring two votes keeps only links supported at both endpoints.
    """

    sessions = tuple(sessions)
    max_gap = int(max_gap)
    min_bridge_support = int(min_bridge_support)
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    if min_bridge_support < 0:
        raise ValueError("min_bridge_support must be non-negative")
    if not sessions:
        return np.zeros((0, 0), dtype=int)
    if len(sessions) == 1:
        return _roi_indices(sessions[0]).reshape(-1, 1)

    links_by_gap = _thresholded_links_by_gap(
        sessions,
        transform_type=transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        max_gap=max_gap,
    )
    supported_links = filter_gap_links_by_bridge_support(
        links_by_gap,
        max_gap=max_gap,
        min_bridge_support=min_bridge_support,
    )
    return tracks_from_gap_links(sessions, supported_links, max_gap=max_gap)


def filter_gap_links_by_bridge_support(
    links_by_gap: Mapping[tuple[int, int], np.ndarray],
    *,
    max_gap: int,
    min_bridge_support: int = TRACK2P_POLICY_SUPPORTED_GAP_DEFAULT_MIN_BRIDGE_SUPPORT,
) -> dict[tuple[int, int], np.ndarray]:
    """Drop direct skip links that lack adjacent endpoint support."""

    max_gap = int(max_gap)
    min_bridge_support = int(min_bridge_support)
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    if min_bridge_support < 0:
        raise ValueError("min_bridge_support must be non-negative")

    filtered = {
        key: _as_link_matrix(value).copy() for key, value in links_by_gap.items()
    }
    if min_bridge_support == 0 or max_gap <= 1:
        return filtered

    for source_session, step in sorted(filtered):
        if int(step) <= 1:
            continue
        links = _as_link_matrix(filtered[(source_session, step)])
        kept = [
            (int(source_roi), int(target_roi))
            for source_roi, target_roi in links
            if _bridge_support_count(
                links_by_gap,
                source_session=int(source_session),
                step=int(step),
                source_roi=int(source_roi),
                target_roi=int(target_roi),
            )
            >= min_bridge_support
        ]
        filtered[(source_session, step)] = _link_matrix(kept)
    return filtered


def run_track2p_policy_supported_gap_component_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int | None = None,
    min_bridge_support: int = TRACK2P_POLICY_SUPPORTED_GAP_DEFAULT_MIN_BRIDGE_SUPPORT,
    cleanup_config: ComponentCleanupConfig | None = None,
    apply_splits: bool = True,
) -> ComponentAuditOutput:
    """Run support-gated gap rescue and split high-risk component bridges."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
        max_gap=(
            TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP
            if max_gap is None
            else max_gap
        ),
    )
    if int(policy_config.max_gap) < 1:
        raise ValueError("max_gap must be at least 1")
    min_bridge_support = int(min_bridge_support)
    if min_bridge_support < 0:
        raise ValueError("min_bridge_support must be non-negative")
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    cleanup_config = cleanup_config or _default_gap_cleanup_config()
    results: list[SubjectBenchmarkResult] = []
    component_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy supported-gap component cleanup requires independent "
                "manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        predicted_full = _normalize_int_track_matrix(
            emulate_track2p_supported_gap_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                max_gap=int(policy_config.max_gap),
                min_bridge_support=min_bridge_support,
            )
        )
        diagnostic_prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=_no_prune_config(),
        )
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        predicted_eval, reference_eval, evaluated_track_ids = (
            _evaluated_prediction_rows(
                predicted_full,
                reference_tracks,
                config=policy_config,
            )
        )
        audit_rows = component_audit_rows(
            predicted_eval,
            reference_eval,
            sessions=sessions,
            diagnostics=diagnostic_prediction.diagnostics,
            subject=subject_dir.name,
            config=cleanup_config,
            track_ids=evaluated_track_ids,
            seed_session=policy_config.seed_session,
        )
        subject_rows = _mark_applied_splits(audit_rows, apply_splits=apply_splits)
        cleaned = (
            apply_weakest_bridge_splits(predicted_full, subject_rows)
            if apply_splits
            else predicted_full
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        scores = _with_scores_metadata(
            scores,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            cleanup_config=cleanup_config,
            max_gap=int(policy_config.max_gap),
            min_bridge_support=min_bridge_support,
            cell_probability_threshold=float(policy_config.cell_probability_threshold),
            transform_type=policy_config.transform_type,
            apply_splits=apply_splits,
            component_rows=subject_rows,
        )
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=(
                    "Track2p-policy supported-gap component cleanup"
                    if apply_splits
                    else "Track2p-policy supported-gap component audit"
                ),
                method=cast(Any, TRACK2P_POLICY_SUPPORTED_GAP_COMPONENT_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        component_rows.extend(
            _with_metadata(
                subject_rows,
                {
                    "threshold_method": str(threshold_method),
                    "iou_distance_threshold": float(iou_distance_threshold),
                    "cell_probability_threshold": float(
                        policy_config.cell_probability_threshold
                    ),
                    "transform_type": str(policy_config.transform_type),
                    "max_gap": int(policy_config.max_gap),
                    "min_bridge_support": int(min_bridge_support),
                    "cleanup_method": TRACK2P_POLICY_SUPPORTED_GAP_COMPONENT_CLEANUP_METHOD,
                },
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(component_rows))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for supported-gap component cleanup."""

    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "bayescatrack.experiments.track2p_policy_supported_gap_component_cleanup"
        ),
        description=(
            "Run Track2p-policy gap rescue with adjacent endpoint-support gating "
            "followed by weakest-bridge component cleanup."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        default="manual-gt",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", choices=("auto", "suite2p", "npy"), default="suite2p"
    )
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    parser.add_argument(
        "--iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument("--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE)
    parser.add_argument(
        "--max-gap",
        type=int,
        default=TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP,
        help="Maximum session jump for support-gated gap rescue before cleanup.",
    )
    parser.add_argument(
        "--min-bridge-support",
        type=int,
        default=TRACK2P_POLICY_SUPPORTED_GAP_DEFAULT_MIN_BRIDGE_SUPPORT,
        help=(
            "Minimum adjacent endpoint-support votes required for direct skip links. "
            "Use 0 to recover raw gap rescue; use 2 to require both endpoints."
        ),
    )
    parser.add_argument(
        "--apply-splits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_REQUIRE_COMPLETE_TRACK,
    )
    parser.add_argument("--threshold-margin-scale", type=float, default=0.10)
    parser.add_argument("--competition-margin-scale", type=float, default=0.20)
    parser.add_argument("--area-ratio-floor", type=float, default=0.45)
    parser.add_argument("--centroid-distance-scale", type=float, default=4.0)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument(
        "--min-side-observations",
        type=int,
        default=TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MIN_SIDE_OBSERVATIONS,
    )
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--component-output", type=Path, default=None)
    parser.add_argument("--component-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy supported-gap component-cleanup CLI."""

    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        threshold_margin_scale=args.threshold_margin_scale,
        competition_margin_scale=args.competition_margin_scale,
        area_ratio_floor=args.area_ratio_floor,
        centroid_distance_scale=args.centroid_distance_scale,
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        plane_name=args.plane_name,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        max_gap=args.max_gap,
        allow_track2p_as_reference_for_smoke_test=(
            args.allow_track2p_as_reference_for_smoke_test
        ),
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    output = run_track2p_policy_supported_gap_component_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        max_gap=int(args.max_gap),
        min_bridge_support=int(args.min_bridge_support),
        cleanup_config=cleanup_config,
        apply_splits=bool(args.apply_splits),
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.component_output is not None:
        write_component_rows(
            output.component_rows,
            args.component_output,
            output_format=cast(Literal["csv", "json"], args.component_format),
        )
    return 0


def _bridge_support_count(
    links_by_gap: Mapping[tuple[int, int], np.ndarray],
    *,
    source_session: int,
    step: int,
    source_roi: int,
    target_roi: int,
) -> int:
    if int(step) <= 1:
        return 2
    support = 0
    left_links = _as_link_matrix(links_by_gap.get((int(source_session), 1)))
    if left_links.size and np.any(left_links[:, 0] == int(source_roi)):
        support += 1
    right_links = _as_link_matrix(
        links_by_gap.get((int(source_session) + int(step) - 1, 1))
    )
    if right_links.size and np.any(right_links[:, 1] == int(target_roi)):
        support += 1
    return support


def _as_link_matrix(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros((0, 2), dtype=int)
    links = np.asarray(value, dtype=int)
    if links.size == 0:
        return np.zeros((0, 2), dtype=int)
    if links.ndim != 2 or links.shape[1] != 2:
        raise ValueError(f"gap links must have shape (n, 2), got {links.shape}")
    return links


def _link_matrix(values: Sequence[tuple[int, int]]) -> np.ndarray:
    if not values:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(values, dtype=int).reshape(-1, 2)


def _with_scores_metadata(
    scores: Mapping[str, float | int | str],
    *,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cleanup_config: ComponentCleanupConfig,
    max_gap: int,
    min_bridge_support: int,
    cell_probability_threshold: float,
    transform_type: str,
    apply_splits: bool,
    component_rows: Sequence[Mapping[str, float | int | str]],
) -> dict[str, float | int | str]:
    candidate_splits = int(
        sum(int(row["would_split_at_weakest_edge"]) for row in component_rows)
    )
    applied_splits = int(
        sum(int(row["applied_split"]) for row in component_rows) if apply_splits else 0
    )
    return {
        **dict(scores),
        "track2p_policy_variant": TRACK2P_POLICY_SUPPORTED_GAP_COMPONENT_CLEANUP_METHOD,
        "track2p_policy_threshold_method": str(threshold_method),
        "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
        "track2p_policy_cell_probability_threshold": float(cell_probability_threshold),
        "track2p_policy_transform_type": str(transform_type),
        "track2p_policy_max_gap": int(max_gap),
        "track2p_policy_min_bridge_support": int(min_bridge_support),
        "track2p_component_apply_splits": int(apply_splits),
        "track2p_component_candidate_splits": candidate_splits,
        "track2p_component_applied_splits": applied_splits,
        "track2p_component_split_risk_threshold": float(
            cleanup_config.split_risk_threshold
        ),
        "track2p_component_split_penalty": float(cleanup_config.split_penalty),
        "track2p_component_min_side_observations": int(
            cleanup_config.min_side_observations
        ),
        "track2p_component_require_complete_track": int(
            cleanup_config.require_complete_track
        ),
    }


def _with_metadata(
    rows: Sequence[Mapping[str, float | int | str]],
    metadata: Mapping[str, Any],
) -> list[dict[str, float | int | str]]:
    formatted = {key: _format_metadata_value(value) for key, value in metadata.items()}
    return [{**dict(row), **formatted} for row in rows]


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
