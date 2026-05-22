"""Run the compact Track2p-policy DP benchmark plan."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

from bayescatrack.experiments.benchmark_comparison import main as compare_main
from bayescatrack.experiments.track2p_benchmark import (
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    write_results,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    run_track2p_policy_benchmark,
)
from bayescatrack.experiments.track2p_policy_dp_benchmark import (
    Track2pPolicyDPConfig,
    run_track2p_policy_dp_benchmark,
)

ACCEPT_COMPLETE_TRACK_F1_MICRO = 0.933333
ACCEPT_PAIRWISE_F1_MICRO = 0.961444


def _base_config(args: argparse.Namespace) -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        transform_type=args.transform_type,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
    )


def _write_result_rows(rows: Sequence[SubjectBenchmarkResult], path: Path) -> None:
    write_results([row.to_dict() for row in rows], path, "csv")


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else float(2 * tp / denom)


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _aggregate(
    label: str,
    rows: Sequence[SubjectBenchmarkResult],
    *,
    parameters: dict[str, float | int] | None = None,
) -> dict[str, float | int | str]:
    row_dicts = [row.to_dict() for row in rows]
    pair_tp = sum(_as_int(row.get("pairwise_true_positives")) for row in row_dicts)
    pair_fp = sum(_as_int(row.get("pairwise_false_positives")) for row in row_dicts)
    pair_fn = sum(_as_int(row.get("pairwise_false_negatives")) for row in row_dicts)
    complete_tp = sum(
        _as_int(row.get("complete_track_true_positives")) for row in row_dicts
    )
    complete_fp = sum(
        _as_int(row.get("complete_track_false_positives")) for row in row_dicts
    )
    complete_fn = sum(
        _as_int(row.get("complete_track_false_negatives")) for row in row_dicts
    )
    pairwise_f1 = [_as_float(row.get("pairwise_f1")) for row in row_dicts]
    complete_f1 = [_as_float(row.get("complete_track_f1")) for row in row_dicts]
    aggregate = {
        "label": label,
        "n_subjects": len(row_dicts),
        "pairwise_f1_mean": _nanmean(pairwise_f1),
        "complete_track_f1_mean": _nanmean(complete_f1),
        "pairwise_f1_micro": _f1(pair_tp, pair_fp, pair_fn),
        "complete_track_f1_micro": _f1(complete_tp, complete_fp, complete_fn),
        "pairwise_true_positives": pair_tp,
        "pairwise_false_positives": pair_fp,
        "pairwise_false_negatives": pair_fn,
        "complete_track_true_positives": complete_tp,
        "complete_track_false_positives": complete_fp,
        "complete_track_false_negatives": complete_fn,
    }
    if parameters:
        aggregate.update(parameters)
    return aggregate


def _nanmean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return math.nan if not finite else float(sum(finite) / len(finite))


def _write_dict_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _compare_core_outputs(output_dir: Path) -> None:
    compare_main(
        [
            "--input",
            f"Track2p-policy min={output_dir / 'track2p_policy_min.csv'}",
            "--input",
            f"DP default={output_dir / 'track2p_policy_dp_default.csv'}",
            "--input",
            f"DP top-k only={output_dir / 'track2p_policy_dp_topk_only.csv'}",
            "--input",
            f"DP conservative gap={output_dir / 'track2p_policy_dp_conservative_gap.csv'}",
            "--input",
            f"DP aggressive={output_dir / 'track2p_policy_dp_aggressive.csv'}",
            "--output",
            str(output_dir / "policy_dp_comparison.md"),
            "--format",
            "markdown",
            "--highlight-best",
        ]
    )


def _dp_config(
    *,
    threshold_method: str,
    iou_distance_threshold: float,
    row_top_k: int,
    rescue_min_iou: float,
    threshold_rescue_margin: float,
    gap_penalty: float,
    beam_width: int,
    max_gap: int,
) -> Track2pPolicyDPConfig:
    return Track2pPolicyDPConfig(
        threshold_method=threshold_method,
        iou_distance_threshold=iou_distance_threshold,
        row_top_k=row_top_k,
        rescue_min_iou=rescue_min_iou,
        threshold_rescue_margin=threshold_rescue_margin,
        gap_penalty=gap_penalty,
        beam_width=beam_width,
        max_gap=max_gap,
    )


def _run_dp(
    config: Track2pBenchmarkConfig,
    *,
    dp_config: Track2pPolicyDPConfig,
    transform_type: str,
    cell_probability_threshold: float,
) -> list[SubjectBenchmarkResult]:
    return run_track2p_policy_dp_benchmark(
        config,
        dp_config=dp_config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )


def _selection_key(row: dict[str, object]) -> tuple[float, float, int, int, float, float]:
    return (
        -_as_float(row["complete_track_f1_micro"]),
        -_as_float(row["pairwise_f1_micro"]),
        _as_int(row["pairwise_false_positives"]),
        _as_int(row["row_top_k"]),
        -_as_float(row["rescue_min_iou"]),
        _as_float(row["threshold_rescue_margin"]),
    )


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _base_config(args)

    policy_rows = run_track2p_policy_benchmark(
        config,
        threshold_method=args.threshold_method,
        iou_distance_threshold=args.iou_distance_threshold,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
    )
    _write_result_rows(policy_rows, output_dir / "track2p_policy_min.csv")

    core_variants = [
        (
            "DP default",
            "track2p_policy_dp_default.csv",
            _dp_config(
                threshold_method=args.threshold_method,
                iou_distance_threshold=args.iou_distance_threshold,
                row_top_k=2,
                rescue_min_iou=0.10,
                threshold_rescue_margin=0.15,
                gap_penalty=1.0,
                beam_width=8,
                max_gap=2,
            ),
        ),
        (
            "DP top-k only",
            "track2p_policy_dp_topk_only.csv",
            _dp_config(
                threshold_method=args.threshold_method,
                iou_distance_threshold=args.iou_distance_threshold,
                row_top_k=2,
                rescue_min_iou=0.10,
                threshold_rescue_margin=0.15,
                gap_penalty=999.0,
                beam_width=8,
                max_gap=1,
            ),
        ),
        (
            "DP conservative gap",
            "track2p_policy_dp_conservative_gap.csv",
            _dp_config(
                threshold_method=args.threshold_method,
                iou_distance_threshold=args.iou_distance_threshold,
                row_top_k=2,
                rescue_min_iou=0.20,
                threshold_rescue_margin=0.05,
                gap_penalty=1.5,
                beam_width=8,
                max_gap=2,
            ),
        ),
        (
            "DP aggressive",
            "track2p_policy_dp_aggressive.csv",
            _dp_config(
                threshold_method=args.threshold_method,
                iou_distance_threshold=args.iou_distance_threshold,
                row_top_k=3,
                rescue_min_iou=0.08,
                threshold_rescue_margin=0.20,
                gap_penalty=0.5,
                beam_width=16,
                max_gap=2,
            ),
        ),
    ]

    aggregate_rows = [_aggregate("Track2p-policy min", policy_rows)]
    default_aggregate: dict[str, float | int | str] | None = None
    for label, filename, dp_config in core_variants:
        rows = _run_dp(
            config,
            dp_config=dp_config,
            transform_type=args.transform_type,
            cell_probability_threshold=args.cell_probability_threshold,
        )
        _write_result_rows(rows, output_dir / filename)
        aggregate = _aggregate(
            label,
            rows,
            parameters={
                "row_top_k": dp_config.row_top_k,
                "rescue_min_iou": dp_config.rescue_min_iou,
                "threshold_rescue_margin": dp_config.threshold_rescue_margin,
                "gap_penalty": dp_config.gap_penalty,
                "beam_width": dp_config.beam_width,
                "max_gap": dp_config.max_gap,
            },
        )
        aggregate_rows.append(aggregate)
        if label == "DP default":
            default_aggregate = aggregate

    _write_dict_rows(output_dir / "policy_dp_core_aggregate.csv", aggregate_rows)
    _compare_core_outputs(output_dir)

    if default_aggregate is None:
        raise RuntimeError("DP default aggregate was not computed.")
    viable = (
        _as_float(default_aggregate["complete_track_f1_micro"])
        >= ACCEPT_COMPLETE_TRACK_F1_MICRO
        and _as_float(default_aggregate["pairwise_f1_micro"])
        >= ACCEPT_PAIRWISE_F1_MICRO
    )
    (output_dir / "policy_dp_viability.txt").write_text(
        (
            "viable\n"
            if viable
            else "not viable; compact 36-setting DP sweep skipped\n"
        ),
        encoding="utf-8",
    )
    if not viable:
        return

    sweep_aggregates: list[dict[str, object]] = []
    sweep_subject_rows: list[dict[str, object]] = []
    best_rows: list[SubjectBenchmarkResult] | None = None
    for row_top_k in (2, 3):
        for rescue_min_iou in (0.10, 0.15, 0.20):
            for threshold_rescue_margin in (0.05, 0.10, 0.15):
                for gap_penalty in (0.75, 1.25):
                    dp_config = _dp_config(
                        threshold_method=args.threshold_method,
                        iou_distance_threshold=args.iou_distance_threshold,
                        row_top_k=row_top_k,
                        rescue_min_iou=rescue_min_iou,
                        threshold_rescue_margin=threshold_rescue_margin,
                        gap_penalty=gap_penalty,
                        beam_width=8,
                        max_gap=2,
                    )
                    rows = _run_dp(
                        config,
                        dp_config=dp_config,
                        transform_type=args.transform_type,
                        cell_probability_threshold=args.cell_probability_threshold,
                    )
                    parameters = {
                        "row_top_k": row_top_k,
                        "rescue_min_iou": rescue_min_iou,
                        "threshold_rescue_margin": threshold_rescue_margin,
                        "gap_penalty": gap_penalty,
                        "beam_width": 8,
                        "max_gap": 2,
                    }
                    label = (
                        f"k{row_top_k}_iou{rescue_min_iou:g}_"
                        f"margin{threshold_rescue_margin:g}_gap{gap_penalty:g}"
                    )
                    aggregate = _aggregate(label, rows, parameters=parameters)
                    sweep_aggregates.append(aggregate)
                    for row in rows:
                        sweep_subject_rows.append(
                            {"setting": label, **parameters, **row.to_dict()}
                        )
                    if min(sweep_aggregates, key=_selection_key) is aggregate:
                        best_rows = rows

    sorted_sweep = sorted(sweep_aggregates, key=_selection_key)
    _write_dict_rows(output_dir / "policy_dp_sweep_aggregate.csv", sorted_sweep)
    _write_dict_rows(output_dir / "policy_dp_sweep_subject_rows.csv", sweep_subject_rows)
    _write_dict_rows(output_dir / "policy_dp_sweep_best.csv", sorted_sweep[:1])
    if best_rows is not None:
        _write_result_rows(best_rows, output_dir / "track2p_policy_dp_best.csv")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--reference-kind",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        default="manual-gt",
    )
    parser.add_argument("--input-format", choices=("auto", "suite2p", "npy"), default="suite2p")
    parser.add_argument("--transform-type", default="affine")
    parser.add_argument("--threshold-method", choices=("otsu", "min"), default="min")
    parser.add_argument("--iou-distance-threshold", type=float, default=12.0)
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--include-non-cells", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=Path("results/policy_dp"))
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
