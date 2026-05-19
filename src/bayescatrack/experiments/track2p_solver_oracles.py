"""Run solver-oracle diagnostics for Track2p-style benchmarks.

The oracles in :mod:`bayescatrack.experiments.solver_oracles` build
pairwise-cost matrices that isolate solver and registration upper bounds.  This
module makes those diagnostics a first-class benchmark command: it runs the
normal PyRecEst multisession solver on each oracle cost matrix, converts the
tracks back to Suite2p ROI indices, and scores them against the same manual-GT
reference path used by the paper-facing Track2p benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.experiments.solver_oracles import (
    oracle_edge_costs,
    oracle_rank_k_costs,
    oracle_registration_costs,
    solve_from_pairwise_costs,
)
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    ProgressReporter,
    ReferenceKind,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    format_benchmark_table,
)
from bayescatrack.reference import Track2pReference

OracleName = Literal["edge-costs", "rank-k", "oracle-registration"]
VALID_ORACLES: tuple[OracleName, ...] = (
    "edge-costs",
    "rank-k",
    "oracle-registration",
)
DEFAULT_RANK_KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class SolverOracleOptions:
    """Non-data options for solver-oracle diagnostics."""

    oracles: tuple[OracleName, ...] = VALID_ORACLES
    rank_ks: tuple[int, ...] = DEFAULT_RANK_KS
    large_cost: float = 1.0e6
    oracle_registration_min_fit_links: int = 3
    oracle_registration_require_full_rank: bool = True
    oracle_registration_ridge: float = 0.0
    continue_on_error: bool = False


@dataclass(frozen=True)
class _SubjectData:
    subject_dir: Path
    sessions: tuple[Any, ...]
    reference: Track2pReference
    reference_matrix: np.ndarray


@dataclass(frozen=True)
class _OracleSpec:
    oracle: OracleName
    variant: str
    rank_k: int | None = None


def run_track2p_solver_oracles(
    config: Track2pBenchmarkConfig,
    *,
    options: SolverOracleOptions | None = None,
    oracles: Sequence[str] | None = None,
    rank_ks: Sequence[int] | None = None,
    large_cost: float | None = None,
    oracle_registration_min_fit_links: int | None = None,
    oracle_registration_require_full_rank: bool | None = None,
    oracle_registration_ridge: float | None = None,
    continue_on_error: bool | None = None,
) -> list[SubjectBenchmarkResult]:
    """Run configured solver-oracle diagnostics and return benchmark rows.

    Parameters are intentionally compatible with the other Track2p benchmark
    harnesses.  The returned rows can be written as CSV/JSON/table and compared
    directly to ordinary ``global-assignment`` rows.
    """

    resolved = _resolve_options(
        options,
        oracles=oracles,
        rank_ks=rank_ks,
        large_cost=large_cost,
        oracle_registration_min_fit_links=oracle_registration_min_fit_links,
        oracle_registration_require_full_rank=oracle_registration_require_full_rank,
        oracle_registration_ridge=oracle_registration_ridge,
        continue_on_error=continue_on_error,
    )
    _validate_config(config)
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    specs = _oracle_specs(resolved)
    progress = ProgressReporter(
        len(subject_dirs), enabled=config.progress, label="solver-oracles"
    )
    for subject_dir in subject_dirs:
        progress.step(f"running {subject_dir.name}")
        subject = _load_subject_data(subject_dir, config)
        for spec in specs:
            try:
                results.append(_run_one_oracle(subject, config, resolved, spec))
            except Exception as exc:  # pragma: no cover - exercised by CLI/users
                if not resolved.continue_on_error:
                    raise
                results.append(_error_result(subject, config, resolved, spec, exc))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for solver-oracle diagnostics."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark solver-oracles",
        description=(
            "Run manual-GT solver-oracle diagnostics through the normal "
            "global-assignment solver."
        ),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument(
        "--oracle",
        dest="oracles",
        action="append",
        choices=VALID_ORACLES,
        default=None,
        help="Oracle diagnostic to run; repeat to select a subset. Defaults to all.",
    )
    parser.add_argument(
        "--rank-ks",
        default=",".join(str(value) for value in DEFAULT_RANK_KS),
        help="Comma-separated k values for the rank-k oracle.",
    )
    parser.add_argument("--large-cost", type=float, default=1.0e6)
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument(
        "--cost",
        default="registered-iou",
        choices=(
            "registered-iou",
            "registered-soft-iou",
            "registered-shifted-iou",
            "roi-aware",
            "roi-aware-shifted",
        ),
        help=(
            "Base cost for rank-k and oracle-registration diagnostics. "
            "Oracle-registration currently supports registered-iou and roi-aware."
        ),
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-translation", "none"),
    )
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument("--no-cost-threshold", action="store_true")
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=True
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
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--pairwise-cost-kwargs-json", default=None)
    parser.add_argument("--oracle-registration-min-fit-links", type=int, default=3)
    parser.add_argument(
        "--oracle-registration-require-full-rank",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--oracle-registration-ridge", type=float, default=0.0)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an error row for failing oracle variants instead of aborting.",
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the solver-oracle CLI."""

    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    options = SolverOracleOptions(
        oracles=_normalize_oracles(args.oracles),
        rank_ks=parse_rank_ks(args.rank_ks),
        large_cost=_positive_float(args.large_cost, name="--large-cost"),
        oracle_registration_min_fit_links=_positive_int(
            args.oracle_registration_min_fit_links,
            name="--oracle-registration-min-fit-links",
        ),
        oracle_registration_require_full_rank=(
            args.oracle_registration_require_full_rank
        ),
        oracle_registration_ridge=_nonnegative_float(
            args.oracle_registration_ridge, name="--oracle-registration-ridge"
        ),
        continue_on_error=bool(args.continue_on_error),
    )
    results = run_track2p_solver_oracles(config, options=options)
    rows = [result.to_dict() for result in results]
    _write_rows(rows, args.output, cast(OutputFormat, args.format))
    return 0


def _run_one_oracle(
    subject: _SubjectData,
    config: Track2pBenchmarkConfig,
    options: SolverOracleOptions,
    spec: _OracleSpec,
) -> SubjectBenchmarkResult:
    pairwise_costs = _build_oracle_costs(subject, config, options, spec)
    assignment = solve_from_pairwise_costs(
        pairwise_costs,
        subject.sessions,
        max_gap=config.max_gap,
        start_cost=config.start_cost,
        end_cost=config.end_cost,
        gap_penalty=config.gap_penalty,
        cost_threshold=config.cost_threshold,
    )
    predicted = tracks_to_suite2p_index_matrix(
        assignment.result.tracks, subject.sessions
    )
    scores: dict[str, float | int | str] = {
        **_score_prediction_against_reference(
            predicted, subject.reference, config=config
        ),
        **_oracle_score_fields(config, options, spec),
    }
    return SubjectBenchmarkResult(
        subject=subject.subject_dir.name,
        variant=spec.variant,
        method=config.method,
        scores=scores,
        n_sessions=subject.reference.n_sessions,
        reference_source=subject.reference.source,
    )


def _build_oracle_costs(
    subject: _SubjectData,
    config: Track2pBenchmarkConfig,
    options: SolverOracleOptions,
    spec: _OracleSpec,
) -> Mapping[tuple[int, int], np.ndarray]:
    if spec.oracle == "edge-costs":
        return oracle_edge_costs(
            subject.sessions,
            subject.reference_matrix,
            max_gap=config.max_gap,
            large_cost=options.large_cost,
        )
    if spec.oracle == "rank-k":
        if spec.rank_k is None:
            raise ValueError("rank-k oracle spec requires rank_k")
        return oracle_rank_k_costs(
            subject.sessions,
            subject.reference_matrix,
            rank_k=spec.rank_k,
            max_gap=config.max_gap,
            cost=config.cost,
            transform_type=config.transform_type,
            order=config.order,
            weighted_centroids=config.weighted_centroids,
            velocity_variance=config.velocity_variance,
            regularization=config.regularization,
            pairwise_cost_kwargs=config.pairwise_cost_kwargs,
            large_cost=options.large_cost,
        )
    if spec.oracle == "oracle-registration":
        return oracle_registration_costs(
            subject.sessions,
            subject.reference_matrix,
            max_gap=config.max_gap,
            cost=config.cost,
            order=config.order,
            weighted_centroids=config.weighted_centroids,
            velocity_variance=config.velocity_variance,
            regularization=config.regularization,
            pairwise_cost_kwargs=config.pairwise_cost_kwargs,
            large_cost=options.large_cost,
            min_fit_links=options.oracle_registration_min_fit_links,
            require_full_rank=options.oracle_registration_require_full_rank,
            ridge=options.oracle_registration_ridge,
        )
    raise ValueError(f"Unsupported solver oracle: {spec.oracle!r}")


def _oracle_specs(options: SolverOracleOptions) -> tuple[_OracleSpec, ...]:
    specs: list[_OracleSpec] = []
    for oracle in options.oracles:
        if oracle == "edge-costs":
            specs.append(
                _OracleSpec(oracle=oracle, variant="Solver oracle: GT edge costs")
            )
        elif oracle == "rank-k":
            for rank_k in options.rank_ks:
                specs.append(
                    _OracleSpec(
                        oracle=oracle,
                        variant=f"Solver oracle: GT edge rank <= {rank_k}",
                        rank_k=rank_k,
                    )
                )
        elif oracle == "oracle-registration":
            specs.append(
                _OracleSpec(
                    oracle=oracle,
                    variant="Solver oracle: GT affine registration",
                )
            )
        else:  # pragma: no cover - guarded by _normalize_oracles
            raise ValueError(f"Unsupported solver oracle: {oracle!r}")
    return tuple(specs)


def _load_subject_data(
    subject_dir: Path, config: Track2pBenchmarkConfig
) -> _SubjectData:
    sessions = tuple(_load_subject_sessions(subject_dir, config))
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    _validate_reference_roi_indices(reference, sessions)
    return _SubjectData(
        subject_dir=subject_dir,
        sessions=sessions,
        reference=reference,
        reference_matrix=_reference_matrix(reference, curated_only=config.curated_only),
    )


def _oracle_score_fields(
    config: Track2pBenchmarkConfig,
    options: SolverOracleOptions,
    spec: _OracleSpec,
) -> dict[str, float | int | str]:
    fields: dict[str, float | int | str] = {
        "oracle": spec.oracle,
        "oracle_base_cost": config.cost,
        "oracle_large_cost": float(options.large_cost),
        "oracle_max_gap": int(config.max_gap),
        "oracle_transform_type": config.transform_type,
        "oracle_solver_start_cost": float(config.start_cost),
        "oracle_solver_end_cost": float(config.end_cost),
        "oracle_solver_gap_penalty": float(config.gap_penalty),
        "oracle_solver_cost_threshold": _threshold_label(config.cost_threshold),
    }
    if spec.rank_k is not None:
        fields["oracle_rank_k"] = int(spec.rank_k)
    if spec.oracle == "oracle-registration":
        fields.update(
            {
                "oracle_registration_min_fit_links": int(
                    options.oracle_registration_min_fit_links
                ),
                "oracle_registration_require_full_rank": int(
                    options.oracle_registration_require_full_rank
                ),
                "oracle_registration_ridge": float(options.oracle_registration_ridge),
            }
        )
    return fields


def _error_result(
    subject: _SubjectData,
    config: Track2pBenchmarkConfig,
    options: SolverOracleOptions,
    spec: _OracleSpec,
    exc: Exception,
) -> SubjectBenchmarkResult:
    scores = {
        **_oracle_score_fields(config, options, spec),
        "oracle_error": f"{type(exc).__name__}: {exc}",
    }
    return SubjectBenchmarkResult(
        subject=subject.subject_dir.name,
        variant=f"{spec.variant} (ERROR)",
        method=config.method,
        scores=scores,
        n_sessions=subject.reference.n_sessions,
        reference_source=subject.reference.source,
    )


def _resolve_options(
    options: SolverOracleOptions | None,
    *,
    oracles: Sequence[str] | None,
    rank_ks: Sequence[int] | None,
    large_cost: float | None,
    oracle_registration_min_fit_links: int | None,
    oracle_registration_require_full_rank: bool | None,
    oracle_registration_ridge: float | None,
    continue_on_error: bool | None,
) -> SolverOracleOptions:
    base = options or SolverOracleOptions()
    return SolverOracleOptions(
        oracles=base.oracles if oracles is None else _normalize_oracles(oracles),
        rank_ks=base.rank_ks if rank_ks is None else _normalize_rank_ks(rank_ks),
        large_cost=(
            base.large_cost
            if large_cost is None
            else _positive_float(large_cost, name="large_cost")
        ),
        oracle_registration_min_fit_links=(
            base.oracle_registration_min_fit_links
            if oracle_registration_min_fit_links is None
            else _positive_int(
                oracle_registration_min_fit_links,
                name="oracle_registration_min_fit_links",
            )
        ),
        oracle_registration_require_full_rank=(
            base.oracle_registration_require_full_rank
            if oracle_registration_require_full_rank is None
            else bool(oracle_registration_require_full_rank)
        ),
        oracle_registration_ridge=(
            base.oracle_registration_ridge
            if oracle_registration_ridge is None
            else _nonnegative_float(
                oracle_registration_ridge, name="oracle_registration_ridge"
            )
        ),
        continue_on_error=(
            base.continue_on_error
            if continue_on_error is None
            else bool(continue_on_error)
        ),
    )


def _validate_config(config: Track2pBenchmarkConfig) -> None:
    if config.method != "global-assignment":
        raise ValueError("Solver-oracle diagnostics require method='global-assignment'")
    if config.split != "subject":
        raise ValueError("Solver-oracle diagnostics currently run per-subject only")
    if config.cost == "calibrated":
        raise ValueError("Solver-oracle diagnostics require a non-calibrated base cost")


def _config_from_args(args: argparse.Namespace) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        parsed = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_cost_kwargs = parsed
    return Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="subject",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=cast(ReferenceKind, args.reference_kind),
        allow_track2p_as_reference_for_smoke_test=(
            args.allow_track2p_as_reference_for_smoke_test
        ),
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
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        progress=args.progress,
    )


def parse_rank_ks(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated positive integer rank-k list."""

    tokens = tuple(token.strip() for token in raw.split(","))
    if not tokens or any(not token for token in tokens):
        raise ValueError("--rank-ks must be a comma-separated list without empties")
    values: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f"--rank-ks contains a non-integer value: {token!r}"
            ) from exc
        values.append(value)
    return _normalize_rank_ks(values)


def _normalize_rank_ks(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(value) for value in values))
    if not result or any(value < 1 for value in result):
        raise ValueError("rank-k values must be positive integers")
    return result


def _normalize_oracles(oracles: Sequence[str] | None) -> tuple[OracleName, ...]:
    if oracles is None:
        return VALID_ORACLES
    values = tuple(dict.fromkeys(str(oracle) for oracle in oracles))
    invalid = sorted(set(values) - set(VALID_ORACLES))
    if invalid:
        raise ValueError(f"Unsupported solver oracle(s): {', '.join(invalid)}")
    if not values:
        raise ValueError("At least one solver oracle must be selected")
    return cast(tuple[OracleName, ...], values)


def _positive_int(value: int, *, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_float(value: float, *, name: str) -> float:
    result = float(value)
    if result <= 0.0 or not np.isfinite(result):
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _nonnegative_float(value: float, *, name: str) -> float:
    result = float(value)
    if result < 0.0 or not np.isfinite(result):
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _threshold_label(threshold: float | None) -> float | str:
    return "none" if threshold is None else float(threshold)


def _write_rows(
    rows: Sequence[Mapping[str, float | int | str]],
    output: Path | None,
    output_format: OutputFormat,
) -> None:
    if output_format == "json":
        text = json.dumps(_jsonable(list(rows)), indent=2) + "\n"
    elif output_format == "csv":
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        text = buffer.getvalue()
    else:
        text = format_benchmark_table(list(rows)) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def _fieldnames(rows: Sequence[Mapping[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "n_sessions",
        "reference_source",
        "oracle",
        "oracle_rank_k",
        "oracle_base_cost",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "complete_tracks",
        "mean_track_length",
        "oracle_error",
    ]
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
