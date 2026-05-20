"""Small protocol-search harness for Track2p benchmark experiments."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    run_track2p_benchmark,
)


@dataclass(frozen=True)
class SearchSetting:
    """One Track2p benchmark protocol setting."""

    cost: str
    transform_type: str
    max_gap: int
    start_cost: float
    end_cost: float
    gap_penalty: float
    cost_threshold: float | None
    pairwise_cost_kwargs: dict[str, Any] | None

    def label(self) -> str:
        threshold = "none" if self.cost_threshold is None else f"{self.cost_threshold:g}"
        return (
            f"cost={self.cost};transform={self.transform_type};max_gap={self.max_gap};"
            f"start={self.start_cost:g};end={self.end_cost:g};gap={self.gap_penalty:g};"
            f"threshold={threshold}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-search",
        description="Run a compact grid search over Track2p global-assignment protocols.",
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument("--input-format", default="auto", choices=("auto", "suite2p", "npy"))
    parser.add_argument("--allow-track2p-as-reference-for-smoke-test", action="store_true")
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument("--restrict-to-reference-seed-rois", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--costs",
        default="registered-iou,registered-soft-iou,roi-aware,roi-aware-shifted",
        help="Comma-separated association cost presets",
    )
    parser.add_argument(
        "--transform-types",
        default="fov-translation,affine,rigid",
        help="Comma-separated registration transform types",
    )
    parser.add_argument("--max-gaps", default="2,3")
    parser.add_argument("--start-costs", default="5")
    parser.add_argument("--end-costs", default="5")
    parser.add_argument("--gap-penalties", default="0.5,1")
    parser.add_argument("--cost-thresholds", default="4,6,8")
    parser.add_argument(
        "--pairwise-cost-kwargs-json",
        default=None,
        help="JSON object merged into all pairwise-cost kwargs",
    )
    parser.add_argument(
        "--adaptive-edge-prior-json",
        default=None,
        help="JSON object configuring ROI-conditioned adaptive edge priors",
    )
    parser.add_argument(
        "--advanced-components",
        action="store_true",
        help="Enable shape, ambiguity-margin and top-k candidate-pruning components",
    )
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument("--exclude-overlapping-pixels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("json", "csv"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    rows = run_search(args)
    if args.output is not None:
        _write_rows(rows, args.output, args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


def run_search(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_pairwise_kwargs = _parse_json_object(args.pairwise_cost_kwargs_json)
    adaptive_edge_prior_config = _parse_json_object(args.adaptive_edge_prior_json)
    rows: list[dict[str, Any]] = []
    settings = list(_iter_settings(args, base_pairwise_kwargs))
    for setting_index, setting in enumerate(settings, start=1):
        config = Track2pBenchmarkConfig(
            data=args.data,
            method="global-assignment",
            plane_name=args.plane_name,
            input_format=args.input_format,
            reference=args.reference,
            reference_kind=args.reference_kind,
            allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
            curated_only=args.curated_only,
            seed_session=args.seed_session,
            restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
            cost=setting.cost,  # type: ignore[arg-type]
            max_gap=setting.max_gap,
            transform_type=setting.transform_type,
            start_cost=setting.start_cost,
            end_cost=setting.end_cost,
            gap_penalty=setting.gap_penalty,
            cost_threshold=setting.cost_threshold,
            include_behavior=args.include_behavior,
            include_non_cells=args.include_non_cells,
            cell_probability_threshold=args.cell_probability_threshold,
            weighted_masks=args.weighted_masks,
            exclude_overlapping_pixels=args.exclude_overlapping_pixels,
            order=args.order,
            weighted_centroids=args.weighted_centroids,
            velocity_variance=args.velocity_variance,
            regularization=args.regularization,
            pairwise_cost_kwargs=setting.pairwise_cost_kwargs,
            adaptive_edge_prior_config=adaptive_edge_prior_config,
            progress=args.progress,
        )
        try:
            results = run_track2p_benchmark(config)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not args.continue_on_error:
                raise
            rows.append(
                {
                    "search_index": setting_index,
                    "search_count": len(settings),
                    "search_setting": setting.label(),
                    "error": type(exc).__name__,
                    "error_message": str(exc),
                    **_setting_row(setting),
                }
            )
            continue
        for result in results:
            rows.append(
                {
                    "search_index": setting_index,
                    "search_count": len(settings),
                    "search_setting": setting.label(),
                    **_setting_row(setting),
                    **result.to_dict(),
                }
            )
    return rows


def _iter_settings(args: argparse.Namespace, base_pairwise_kwargs: dict[str, Any] | None):
    costs = _parse_str_list(args.costs)
    transforms = _parse_str_list(args.transform_types)
    max_gaps = _parse_int_list(args.max_gaps)
    start_costs = _parse_float_list(args.start_costs)
    end_costs = _parse_float_list(args.end_costs)
    gap_penalties = _parse_float_list(args.gap_penalties)
    thresholds = _parse_thresholds(args.cost_thresholds)
    for cost, transform_type, max_gap, start_cost, end_cost, gap_penalty, threshold in itertools.product(
        costs,
        transforms,
        max_gaps,
        start_costs,
        end_costs,
        gap_penalties,
        thresholds,
    ):
        pairwise_kwargs = dict(base_pairwise_kwargs or {})
        if args.advanced_components:
            pairwise_kwargs.update(
                {
                    "shape_descriptor_components": True,
                    "radial_profile_weight": 0.1,
                    "orientation_weight": 0.05,
                    "eccentricity_weight": 0.05,
                    "compactness_weight": 0.05,
                    "ambiguity_margin_components": True,
                    "ambiguity_margin_weight": 0.05,
                    "candidate_top_k_per_roi": 30,
                    "candidate_gate_margin": 4.0,
                }
            )
        yield SearchSetting(
            cost=cost,
            transform_type=transform_type,
            max_gap=int(max_gap),
            start_cost=float(start_cost),
            end_cost=float(end_cost),
            gap_penalty=float(gap_penalty),
            cost_threshold=threshold,
            pairwise_cost_kwargs=pairwise_kwargs or None,
        )


def _setting_row(setting: SearchSetting) -> dict[str, Any]:
    return {
        "search_cost": setting.cost,
        "search_transform_type": setting.transform_type,
        "search_max_gap": setting.max_gap,
        "search_start_cost": setting.start_cost,
        "search_end_cost": setting.end_cost,
        "search_gap_penalty": setting.gap_penalty,
        "search_cost_threshold": "none" if setting.cost_threshold is None else setting.cost_threshold,
        "search_pairwise_cost_kwargs": json.dumps(setting.pairwise_cost_kwargs or {}, sort_keys=True),
    }


def _parse_json_object(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
    return parsed


def _parse_str_list(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in _parse_str_list(raw))


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(value) for value in _parse_str_list(raw))


def _parse_thresholds(raw: str) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for token in _parse_str_list(raw):
        if token.casefold() in {"none", "null", "off", "disabled"}:
            values.append(None)
        else:
            values.append(float(token))
    return tuple(values)


def _write_rows(rows: list[dict[str, Any]], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def _write_stdout(rows: list[dict[str, Any]], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(rows, indent=2))
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=_fieldnames(rows))
    writer.writeheader()
    writer.writerows(rows)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "search_index",
        "search_count",
        "search_setting",
        "subject",
        "variant",
        "pairwise_f1",
        "complete_track_f1",
        "search_cost",
        "search_transform_type",
        "search_max_gap",
        "search_start_cost",
        "search_end_cost",
        "search_gap_penalty",
        "search_cost_threshold",
        "error",
        "error_message",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
