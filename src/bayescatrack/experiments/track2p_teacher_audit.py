"""Track2p teacher/debug-oracle audit for manual-GT benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import AssociationCost
from bayescatrack.experiments.track2p_benchmark import (
    BenchmarkMethod,
    ReferenceKind,
    Track2pBenchmarkConfig,
)

PairMode = Literal["all", "consecutive", "max-gap"]
OutputFormat = Literal["table", "json", "csv"]
InputFormat = Literal["auto", "suite2p", "npy"]
EdgeKey = tuple[int, int, int, int]

TRACK2P_TEACHER_MISS_CATEGORY = "GT+Track2p+Bayes-"
_FIELDS = {
    "GT+Track2p+Bayes+": "edges_gt_track2p_bayes",
    TRACK2P_TEACHER_MISS_CATEGORY: "edges_gt_track2p_not_bayes",
    "GT+Track2p-Bayes+": "edges_gt_not_track2p_bayes",
    "GT+Track2p-Bayes-": "edges_gt_not_track2p_not_bayes",
    "GT-Track2p+Bayes+": "edges_not_gt_track2p_bayes",
    "GT-Track2p+Bayes-": "edges_not_gt_track2p_not_bayes",
    "GT-Track2p-Bayes+": "edges_not_gt_not_track2p_bayes",
}


@dataclass(frozen=True)
class Track2pTeacherAuditConfig:
    data: Path
    ground_truth_reference: Path | None = None
    track2p_reference: Path | None = None
    pair_mode: PairMode = "all"
    plane_name: str = "plane0"
    input_format: InputFormat = "auto"
    curated_only: bool = False
    seed_session: int = 0
    restrict_to_reference_seed_rois: bool = True
    cost: AssociationCost = "registered-iou"
    max_gap: int = 2
    transform_type: str = "affine"
    start_cost: float = 5.0
    end_cost: float = 5.0
    gap_penalty: float = 1.0
    cost_threshold: float | None = 6.0
    include_behavior: bool = True
    include_non_cells: bool = False
    cell_probability_threshold: float = 0.5
    weighted_masks: bool = False
    exclude_overlapping_pixels: bool = True
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1e-6
    pairwise_cost_kwargs: dict[str, Any] | None = None
    progress: bool = False


@dataclass(frozen=True)
class Track2pTeacherAuditResult:
    summary_rows: list[dict[str, Any]]
    edge_rows: list[dict[str, Any]]


def audit_track_matrices(
    *,
    subject: str,
    session_names: Sequence[str],
    ground_truth_tracks: Any,
    track2p_tracks: Any,
    bayes_tracks: Any,
    pair_mode: PairMode = "all",
    max_gap: int = 2,
    seed_session: int = 0,
    restrict_to_reference_seed_rois: bool = True,
) -> Track2pTeacherAuditResult:
    names = tuple(map(str, session_names))
    ground_truth = _normalize_track_matrix(ground_truth_tracks)
    track2p = _normalize_track_matrix(track2p_tracks)
    bayes = _normalize_track_matrix(bayes_tracks)

    for label, matrix in (
        ("ground_truth_tracks", ground_truth),
        ("track2p_tracks", track2p),
        ("bayes_tracks", bayes),
    ):
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError(f"{label} must have one column per session")

    seed_rois = {
        int(value) for value in ground_truth[:, seed_session] if value is not None
    }
    if restrict_to_reference_seed_rois:
        ground_truth = _seed_filter(ground_truth, seed_rois, seed_session)
        track2p = _seed_filter(track2p, seed_rois, seed_session)
        bayes = _seed_filter(bayes, seed_rois, seed_session)

    pairs = _session_pairs(len(names), pair_mode, max_gap)
    ground_truth_edges = _edge_map(ground_truth, pairs)
    track2p_edges = _edge_map(track2p, pairs)
    bayes_edges = _edge_map(bayes, pairs)
    edge_rows = _edge_rows(
        subject, names, ground_truth_edges, track2p_edges, bayes_edges
    )
    summary = _summary(
        subject,
        len(names),
        pair_mode,
        max_gap,
        seed_session,
        len(seed_rois),
        restrict_to_reference_seed_rois,
        set(ground_truth_edges),
        set(track2p_edges),
        set(bayes_edges),
        edge_rows,
    )
    return Track2pTeacherAuditResult([summary], edge_rows)


def run_track2p_teacher_audit(
    config: Track2pTeacherAuditConfig,
) -> Track2pTeacherAuditResult:
    from bayescatrack.experiments.track2p_benchmark import (  # pylint: disable=protected-access
        GROUND_TRUTH_REFERENCE_SOURCE,
        _load_reference_for_subject,
        _load_subject_sessions,
        _predict_subject_tracks,
        _reference_matrix,
        _validate_reference_roi_indices,
        discover_subject_dirs,
    )

    summaries: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    progress = _Progress(len(subject_dirs), config.progress)
    for subject_dir in subject_dirs:
        progress.step(f"auditing {subject_dir.name}")
        ground_truth_config = _bench_cfg(
            config,
            method="global-assignment",
            reference=config.ground_truth_reference,
            reference_kind="manual-gt",
        )
        ground_truth_reference = _load_reference_for_subject(
            subject_dir,
            data_root=config.data,
            config=ground_truth_config,
        )
        if ground_truth_reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError("teacher-audit requires independent manual ground truth")

        _validate_reference_roi_indices(
            ground_truth_reference,
            _load_subject_sessions(subject_dir, ground_truth_config),
        )
        track2p_config = _bench_cfg(
            config,
            method="track2p-baseline",
            reference=config.track2p_reference,
            reference_kind="track2p-output",
            allow_track2p_as_reference_for_smoke_test=True,
        )
        track2p_reference = _load_reference_for_subject(
            subject_dir,
            data_root=config.data,
            config=track2p_config,
        )
        bayes_matrix, _variant = _predict_subject_tracks(
            subject_dir,
            ground_truth_config,
            reference=ground_truth_reference,
        )
        result = audit_track_matrices(
            subject=subject_dir.name,
            session_names=ground_truth_reference.session_names,
            ground_truth_tracks=_reference_matrix(
                ground_truth_reference,
                curated_only=config.curated_only,
            ),
            track2p_tracks=_reference_matrix(
                track2p_reference,
                curated_only=config.curated_only
                and track2p_reference.curated_mask is not None,
            ),
            bayes_tracks=bayes_matrix,
            pair_mode=config.pair_mode,
            max_gap=config.max_gap,
            seed_session=config.seed_session,
            restrict_to_reference_seed_rois=config.restrict_to_reference_seed_rois,
        )
        summaries.extend(result.summary_rows)
        edges.extend(result.edge_rows)
    return Track2pTeacherAuditResult(summaries, edges)


def teacher_training_rows(
    edge_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "subject": row["subject"],
            "session_a": row["session_a"],
            "session_b": row["session_b"],
            "gap": row["gap"],
            "roi_a": row["roi_a"],
            "roi_b": row["roi_b"],
            "teacher_label": int(bool(row["in_track2p"])),
            "manual_gt_label": int(bool(row["in_ground_truth"])),
            "bayes_label": int(bool(row["in_bayes"])),
            "teacher_label_source": "track2p_output",
            "category": row["category"],
        }
        for row in edge_rows
    ]


def format_teacher_audit_table(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        "subject",
        "pair_mode",
        "ground_truth_edges",
        "track2p_edges",
        "bayes_edges",
        "edges_gt_track2p_not_bayes",
        "track2p_vs_gt_f1",
        "bayes_vs_gt_f1",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_fmt(row.get(column, "")) for column in columns) + " |"
        )
    return "\n".join(lines)


def write_edge_rows(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_path)


def write_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
    elif output_format == "csv":
        _write_csv(rows, output_path)
    else:
        output_path.write_text(
            format_teacher_audit_table(rows) + "\n", encoding="utf-8"
        )


# jscpd:ignore-start
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-teacher-audit"
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--ground-truth-reference", type=Path)
    parser.add_argument("--track2p-reference", type=Path)
    parser.add_argument(
        "--pair-mode",
        choices=("all", "consecutive", "max-gap"),
        default="all",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=("auto", "suite2p", "npy"),
    )
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cost",
        default="registered-iou",
        choices=("registered-iou", "roi-aware"),
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--transform-type", default="affine")
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument("--no-cost-threshold", action="store_true")
    parser.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--pairwise-cost-kwargs-json")
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--edges-output", type=Path)
    parser.add_argument("--focus-output", type=Path)
    parser.add_argument("--teacher-output", type=Path)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_track2p_teacher_audit(_config(args))
    output_format = cast(OutputFormat, args.format)
    if args.output:
        write_summary_rows(result.summary_rows, args.output, output_format)
    else:
        _stdout(result.summary_rows, output_format)
    if args.edges_output:
        write_edge_rows(result.edge_rows, args.edges_output)
    if args.focus_output:
        write_edge_rows(
            [
                row
                for row in result.edge_rows
                if row["category"] == TRACK2P_TEACHER_MISS_CATEGORY
            ],
            args.focus_output,
        )
    if args.teacher_output:
        write_edge_rows(teacher_training_rows(result.edge_rows), args.teacher_output)
    return 0


def _edge_rows(
    subject: str,
    names: Sequence[str],
    ground_truth: Mapping[EdgeKey, int],
    track2p: Mapping[EdgeKey, int],
    bayes: Mapping[EdgeKey, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge in sorted(set(ground_truth) | set(track2p) | set(bayes)):
        session_a, session_b, roi_a, roi_b = edge
        in_ground_truth = edge in ground_truth
        in_track2p = edge in track2p
        in_bayes = edge in bayes
        rows.append(
            {
                "subject": subject,
                "session_a": session_a,
                "session_b": session_b,
                "session_a_name": names[session_a],
                "session_b_name": names[session_b],
                "gap": session_b - session_a,
                "roi_a": roi_a,
                "roi_b": roi_b,
                "in_ground_truth": in_ground_truth,
                "in_track2p": in_track2p,
                "in_bayes": in_bayes,
                "category": _category(in_ground_truth, in_track2p, in_bayes),
                "ground_truth_track_row": ground_truth.get(edge, ""),
                "track2p_track_row": track2p.get(edge, ""),
                "bayes_track_row": bayes.get(edge, ""),
            }
        )
    return rows


def _summary(
    subject: str,
    n_sessions: int,
    mode: PairMode,
    max_gap: int,
    seed_session: int,
    n_seed_rois: int,
    restrict_to_reference_seed_rois: bool,
    ground_truth: set[EdgeKey],
    track2p: set[EdgeKey],
    bayes: set[EdgeKey],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter(str(row["category"]) for row in rows)
    track2p_precision, track2p_recall, track2p_f1 = _scores(track2p, ground_truth)
    bayes_precision, bayes_recall, bayes_f1 = _scores(bayes, ground_truth)
    out: dict[str, Any] = {
        "subject": subject,
        "n_sessions": n_sessions,
        "pair_mode": mode,
        "max_gap": max_gap,
        "seed_session": seed_session,
        "reference_seed_rois": n_seed_rois,
        "restrict_to_reference_seed_rois": int(restrict_to_reference_seed_rois),
        "ground_truth_edges": len(ground_truth),
        "track2p_edges": len(track2p),
        "bayes_edges": len(bayes),
        "track2p_vs_gt_precision": track2p_precision,
        "track2p_vs_gt_recall": track2p_recall,
        "track2p_vs_gt_f1": track2p_f1,
        "bayes_vs_gt_precision": bayes_precision,
        "bayes_vs_gt_recall": bayes_recall,
        "bayes_vs_gt_f1": bayes_f1,
    }
    for category, field_name in _FIELDS.items():
        out[field_name] = int(counts.get(category, 0))

    found_by_track2p_and_bayes = int(out["edges_gt_track2p_bayes"])
    missed_by_bayes = int(out["edges_gt_track2p_not_bayes"])
    out["bayes_miss_rate_on_gt_track2p_agreement"] = _ratio(
        missed_by_bayes,
        found_by_track2p_and_bayes + missed_by_bayes,
    )
    return out


def _edge_map(
    matrix: np.ndarray, pairs: Sequence[tuple[int, int]]
) -> dict[EdgeKey, int]:
    out: dict[EdgeKey, int] = {}
    for row_index, row in enumerate(matrix):
        for session_a, session_b in pairs:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if roi_a is None or roi_b is None:
                continue
            out.setdefault((session_a, session_b, int(roi_a), int(roi_b)), row_index)
    return out


def _session_pairs(
    n_sessions: int,
    mode: PairMode,
    max_gap: int,
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for session_a in range(max(0, n_sessions - 1)):
        for session_b in range(session_a + 1, n_sessions):
            gap = session_b - session_a
            if mode == "consecutive" and gap != 1:
                continue
            if mode == "max-gap" and gap > max_gap:
                continue
            pairs.append((session_a, session_b))
    return tuple(pairs)


def _normalize_track_matrix(track_matrix: Any) -> np.ndarray:
    values = np.asarray(track_matrix, dtype=object)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("track matrices must be two-dimensional")

    out = np.empty(values.shape, dtype=object)
    for index, value in np.ndenumerate(values):
        out[index] = _parse_roi(value)
    return out


def _parse_roi(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"", "none", "nan", "null"}:
            return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    try:
        roi = int(value)
    except (TypeError, ValueError):
        return None
    return roi if roi >= 0 else None


def _seed_filter(
    matrix: np.ndarray, seed_rois: set[int], seed_session: int
) -> np.ndarray:
    if not seed_rois:
        return matrix[:0]
    keep = [
        value is not None and int(value) in seed_rois
        for value in matrix[:, seed_session]
    ]
    return matrix[np.asarray(keep, dtype=bool)]


def _scores(
    predicted: set[EdgeKey], reference: set[EdgeKey]
) -> tuple[float, float, float]:
    true_positives = len(predicted & reference)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def _bench_cfg(
    config: Track2pTeacherAuditConfig,
    *,
    method: BenchmarkMethod,
    reference: Path | None,
    reference_kind: ReferenceKind,
    allow_track2p_as_reference_for_smoke_test: bool = False,
) -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(
        data=config.data,
        method=method,
        plane_name=config.plane_name,
        input_format=config.input_format,
        reference=reference,
        reference_kind=reference_kind,
        allow_track2p_as_reference_for_smoke_test=allow_track2p_as_reference_for_smoke_test,
        curated_only=config.curated_only,
        seed_session=config.seed_session,
        restrict_to_reference_seed_rois=config.restrict_to_reference_seed_rois,
        cost=config.cost,
        max_gap=config.max_gap,
        transform_type=config.transform_type,
        start_cost=config.start_cost,
        end_cost=config.end_cost,
        gap_penalty=config.gap_penalty,
        cost_threshold=config.cost_threshold,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
        progress=config.progress,
    )


def _config(args: argparse.Namespace) -> Track2pTeacherAuditConfig:
    pairwise_kwargs: dict[str, Any] | None = None
    if args.pairwise_cost_kwargs_json:
        loaded = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(loaded, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_kwargs = cast(dict[str, Any], loaded)

    return Track2pTeacherAuditConfig(
        data=args.data,
        ground_truth_reference=args.ground_truth_reference,
        track2p_reference=args.track2p_reference,
        pair_mode=cast(PairMode, args.pair_mode),
        plane_name=args.plane_name,
        input_format=cast(InputFormat, args.input_format),
        curated_only=args.curated_only,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        cost=cast(AssociationCost, args.cost),
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        start_cost=args.start_cost,
        end_cost=args.end_cost,
        gap_penalty=args.gap_penalty,
        cost_threshold=None if args.no_cost_threshold else args.cost_threshold,
        include_behavior=args.include_behavior,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        pairwise_cost_kwargs=pairwise_kwargs,
        progress=args.progress,
    )
# jscpd:ignore-end


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fields(rows))
        writer.writeheader()
        writer.writerows(rows)


def _stdout(rows: Sequence[Mapping[str, Any]], output_format: OutputFormat) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
    elif output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_fields(rows))
        writer.writeheader()
        writer.writerows(rows)
    else:
        print(format_teacher_audit_table(rows))


def _fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(key for row in rows for key in row))


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)


def _category(in_ground_truth: bool, in_track2p: bool, in_bayes: bool) -> str:
    return f"GT{'+' if in_ground_truth else '-'}Track2p{'+' if in_track2p else '-'}Bayes{'+' if in_bayes else '-'}"


class _Progress:
    def __init__(self, total: int, enabled: bool) -> None:
        self.total = max(total, 1)
        self.enabled = enabled
        self.current = 0

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        self.current += 1
        print(
            f"teacher-audit {self.current}/{self.total} {message}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
