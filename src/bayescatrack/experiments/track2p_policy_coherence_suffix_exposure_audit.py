"""Audit coherence suffix-gate exposure without manual ground truth.

This diagnostic runs the Track2pPolicy + ComponentCleanup candidate generator
over every Track2p-style subject under the data root and reports only how often
the coherence suffix gate fires. It intentionally does not load manual GT,
score predictions, or emit GT labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _load_subject_sessions,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_coherence_suffix_stitch_whatif import (
    CoherenceSuffixStitchGate,
    _positive_int_arg,
    _positive_int_value,
    _select_paths,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    _mark_applied_splits,
    _normalize_int_track_matrix,
    apply_weakest_bridge_splits,
    component_audit_rows,
)
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)
from bayescatrack.experiments.track2p_policy_suffix_stitch_ranking_audit import (
    _FeatureCache,
    _PathCandidate,
    _ranked_suffix_paths,
)

TRACK2P_POLICY_COHERENCE_SUFFIX_EXPOSURE_AUDIT_METHOD = (
    "track2p-policy-coherence-suffix-exposure-audit"
)


def run_track2p_policy_coherence_suffix_exposure_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    gate: CoherenceSuffixStitchGate | None = None,
    edge_top_k: int = 25,
    path_beam_width: int = 100,
) -> tuple[dict[str, int | str], ...]:
    """Return per-subject suffix-gate exposure rows without loading GT."""

    edge_top_k = _positive_int_value(edge_top_k, name="edge_top_k")
    path_beam_width = _positive_int_value(path_beam_width, name="path_beam_width")
    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    cleanup_config = cleanup_config or ComponentCleanupConfig()
    gate = gate or CoherenceSuffixStitchGate()
    rows: list[dict[str, int | str]] = []
    for subject_dir in subject_dirs:
        paths, selected = _subject_paths_and_selected(
            subject_dir,
            config=policy_config,
            cleanup_config=cleanup_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            gate=gate,
            edge_top_k=edge_top_k,
            path_beam_width=path_beam_width,
        )
        rows.append(_exposure_row(subject_dir.name, paths, selected))
    rows.append(_aggregate_row(rows))
    return tuple(rows)


def _subject_paths_and_selected(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    gate: CoherenceSuffixStitchGate,
    edge_top_k: int,
    path_beam_width: int,
) -> tuple[tuple[_PathCandidate, ...], tuple[_PathCandidate, ...]]:
    sessions = _load_subject_sessions(subject_dir, config)
    prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    policy_full = _normalize_int_track_matrix(prediction.tracks)
    empty_reference = np.empty((0, policy_full.shape[1]), dtype=int)
    audit_rows = component_audit_rows(
        policy_full,
        empty_reference,
        sessions=sessions,
        diagnostics=prediction.diagnostics,
        subject=subject_dir.name,
        config=cleanup_config,
        track_ids=tuple(range(policy_full.shape[0])),
        seed_session=config.seed_session,
    )
    cleaned = apply_weakest_bridge_splits(
        policy_full, _mark_applied_splits(audit_rows, apply_splits=True)
    )
    feature_cache = _FeatureCache(
        sessions=sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(config.cell_probability_threshold),
        matrices={},
    )
    paths = _ranked_suffix_paths(
        cleaned,
        empty_reference,
        subject=subject_dir.name,
        feature_cache=feature_cache,
        max_suffix_length=int(gate.suffix_path_length),
        edge_top_k=int(edge_top_k),
        path_beam_width=int(path_beam_width),
    )
    selected = _select_paths(paths, cleaned, gate=gate)
    return paths, selected


def _exposure_row(
    subject: str,
    paths: Sequence[_PathCandidate],
    selected: Sequence[_PathCandidate],
) -> dict[str, int | str]:
    suffix_fragment_ids = {int(path.component_id) for path in paths}
    selected_lengths = tuple(sorted(int(len(path.edges)) for path in selected))
    return {
        "subject": subject,
        "n_suffix_fragments": int(len(suffix_fragment_ids)),
        "n_candidate_paths": int(len(paths)),
        "n_selected_stitches": int(len(selected)),
        "selected_path_lengths": _int_list(selected_lengths),
        "selected_stitches_per_subject": int(len(selected)),
    }


def _aggregate_row(rows: Sequence[Mapping[str, int | str]]) -> dict[str, int | str]:
    per_subject = tuple(row for row in rows if str(row.get("subject", "")) != "ALL")
    return {
        "subject": "ALL",
        "n_suffix_fragments": int(
            sum(int(row.get("n_suffix_fragments", 0)) for row in per_subject)
        ),
        "n_candidate_paths": int(
            sum(int(row.get("n_candidate_paths", 0)) for row in per_subject)
        ),
        "n_selected_stitches": int(
            sum(int(row.get("n_selected_stitches", 0)) for row in per_subject)
        ),
        "selected_path_lengths": ";".join(
            str(row.get("selected_path_lengths", ""))
            for row in per_subject
            if str(row.get("selected_path_lengths", ""))
        ),
        "selected_stitches_per_subject": ";".join(
            f"{row.get('subject')}:{int(row.get('n_selected_stitches', 0))}"
            for row in per_subject
        ),
    }


def _int_list(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write rows as CSV or JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    fieldnames = (
        "subject",
        "n_suffix_fragments",
        "n_candidate_paths",
        "n_selected_stitches",
        "selected_path_lengths",
        "selected_stitches_per_subject",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the exposure audit."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-coherence-suffix-exposure-audit",
        description=(
            "Run the coherence suffix gate over all Track2p-style subjects "
            "without loading manual GT and report gate-fire frequency."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
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
    parser.add_argument(
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--suffix-path-length", type=_positive_int_arg, default=2)
    parser.add_argument("--min-cell-probability", type=float, default=0.80)
    parser.add_argument("--min-area-ratio", type=float, default=0.80)
    parser.add_argument("--max-centroid-distance", type=float, default=6.0)
    parser.add_argument("--min-shifted-iou", type=float, default=0.30)
    parser.add_argument("--min-motion-consistency", type=float, default=0.50)
    parser.add_argument("--min-shape-consistency", type=float, default=0.82)
    parser.add_argument("--max-stitches-per-subject", type=_positive_int_arg, default=1)
    parser.add_argument("--edge-top-k", type=_positive_int_arg, default=25)
    parser.add_argument("--path-beam-width", type=_positive_int_arg, default=100)
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--aggregate-row",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include an ALL aggregate row in the exposure output.",
    )
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    """Run the no-GT exposure audit CLI."""

    args = (parser or build_arg_parser()).parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    gate = CoherenceSuffixStitchGate(
        suffix_path_length=int(args.suffix_path_length),
        min_cell_probability=float(args.min_cell_probability),
        min_area_ratio=float(args.min_area_ratio),
        max_centroid_distance=float(args.max_centroid_distance),
        min_shifted_iou=float(args.min_shifted_iou),
        min_motion_consistency=float(args.min_motion_consistency),
        min_shape_consistency=float(args.min_shape_consistency),
        max_stitches_per_subject=int(args.max_stitches_per_subject),
    )
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        plane_name=args.plane_name,
        seed_session=args.seed_session,
        transform_type=args.transform_type,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    rows = run_track2p_policy_coherence_suffix_exposure_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
        gate=gate,
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
    )
    output_rows = (
        rows
        if bool(args.aggregate_row)
        else tuple(row for row in rows if str(row.get("subject")) != "ALL")
    )
    write_rows(output_rows, args.output, output_format=args.format)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
