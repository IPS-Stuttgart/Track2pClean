"""LOSO solver-prior tuning for calibrated Track2p global assignment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    collect_reference_pairwise_example_blocks,
    collect_reference_training_examples,
    fit_logistic_association_model,
)
from bayescatrack.association.monotone_ranker import (
    MonotoneRankerOptions,
    fit_monotone_ranking_association_model_from_blocks,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    build_registered_pairwise_costs,
    session_edge_pairs,
    solve_global_assignment_from_pairwise_costs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.evaluation.calibration_diagnostics import calibration_summary
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    format_benchmark_table,
    _score_prediction_against_reference,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    LosoCalibrationFold,
    LosoCalibrationResult,
    SubjectCalibrationData,
    _collect_training_examples,
    _load_subject_calibration_data,
    _loso_logistic_model_kwargs,
    _reference_training_options,
    _stringify_class_weight,
    _training_sample_weight,
    _validate_sample_weight_strategy,
)

CalibrationModelKind = Literal["logistic", "monotone"]
SampleWeightStrategy = Literal["none", "balanced"]
SolverPriorObjective = Literal["pairwise_f1", "complete_track_f1"]

DEFAULT_SOLVER_PRIOR_START_COSTS = (1.0, 2.0, 5.0)
DEFAULT_SOLVER_PRIOR_END_COSTS = (1.0, 2.0, 5.0)
DEFAULT_SOLVER_PRIOR_GAP_PENALTIES = (0.6, 1.0, 0.0)
DEFAULT_SOLVER_PRIOR_COST_THRESHOLDS = (2.0, 4.0, 6.0, None)


@dataclass(frozen=True)
class SolverPriorParameters:
    """Global assignment prior constants selected inside a training fold."""

    start_cost: float
    end_cost: float
    gap_penalty: float
    cost_threshold: float | None


@dataclass(frozen=True)
class SolverPriorTuningOptions:
    """Candidate grid and objective for fold-internal solver-prior tuning."""

    objective: SolverPriorObjective = "pairwise_f1"
    start_costs: tuple[float, ...] | None = None
    end_costs: tuple[float, ...] | None = None
    gap_penalties: tuple[float, ...] | None = None
    cost_thresholds: tuple[float | None, ...] | None = None


@dataclass(frozen=True)
class SolverPriorTuningResult:
    """Best solver-prior setting and mean training-subject scores."""

    parameters: SolverPriorParameters
    objective_name: SolverPriorObjective
    objective_value: float
    candidate_count: int
    mean_scores: Mapping[str, float]

    def to_score_dict(self) -> dict[str, float | int | str]:
        return {
            "solver_prior_objective": self.objective_name,
            "solver_prior_objective_score": float(self.objective_value),
            "solver_prior_candidate_count": int(self.candidate_count),
            "tuned_start_cost": float(self.parameters.start_cost),
            "tuned_end_cost": float(self.parameters.end_cost),
            "tuned_gap_penalty": float(self.parameters.gap_penalty),
            "tuned_cost_threshold": _threshold_label(self.parameters.cost_threshold),
        }


@dataclass(frozen=True)
class CachedSolverTuningSubject:
    """Training subject with expensive registered pairwise costs cached once."""

    subject: SubjectCalibrationData
    pairwise_costs: Mapping[tuple[int, int], np.ndarray]
    session_sizes: tuple[int, ...]
    session_edges: tuple[tuple[int, int], ...]


# pylint: disable=too-many-arguments,too-many-locals
def run_track2p_loso_solver_prior_tuning(
    config: Track2pBenchmarkConfig,
    *,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    sample_weight: Any | None = None,
    sample_weight_strategy: SampleWeightStrategy = "none",
    model_kwargs: Mapping[str, Any] | None = None,
    calibration_model: CalibrationModelKind = "logistic",
    monotone_options: MonotoneRankerOptions | None = None,
    solver_prior_options: SolverPriorTuningOptions | None = None,
) -> LosoCalibrationResult:
    """Run calibrated LOSO while tuning solver priors on training subjects.

    This experiment keeps classifier fitting and held-out evaluation identical to
    the calibrated LOSO benchmark, but selects the global assignment start/end,
    gap, and threshold constants using only the training subjects in each fold.
    """

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "LOSO solver-prior tuning requires method='global-assignment' and cost='calibrated'"
        )
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError(
            "LOSO solver-prior tuning requires at least two subject directories"
        )

    feature_names = tuple(feature_names)
    sample_weight_strategy = _validate_sample_weight_strategy(sample_weight_strategy)
    logistic_model_kwargs = _loso_logistic_model_kwargs(model_kwargs)
    if calibration_model not in {"logistic", "monotone"}:
        raise ValueError("calibration_model must be either 'logistic' or 'monotone'")
    monotone_options = monotone_options or MonotoneRankerOptions()
    solver_prior_options = solver_prior_options or SolverPriorTuningOptions()
    progress = ProgressReporter(
        len(subject_dirs) + len(subject_dirs) * (len(subject_dirs) + 3),
        enabled=config.progress,
        label="LOSO-priors",
    )

    subjects = []
    for subject_dir in subject_dirs:
        progress.step(f"loading {subject_dir.name}")
        subjects.append(_load_subject_calibration_data(subject_dir, config=config))
    subject_data = tuple(subjects)
    folds: list[LosoCalibrationFold] = []

    for held_out_index, held_out in enumerate(subject_data):
        training_subjects = tuple(
            subject
            for index, subject in enumerate(subject_data)
            if index != held_out_index
        )
        calibrated_model, training_scores = _fit_calibrated_cost_model(
            training_subjects,
            config=config,
            feature_names=feature_names,
            calibration_model=calibration_model,
            sample_weight=sample_weight,
            sample_weight_strategy=sample_weight_strategy,
            logistic_model_kwargs=logistic_model_kwargs,
            monotone_options=monotone_options,
            progress=progress,
            held_out_subject=held_out.subject_name,
        )
        progress.step(f"tuning solver priors for {held_out.subject_name}")
        tuning_result = tune_solver_priors_for_training_subjects(
            training_subjects,
            config=config,
            cost="calibrated",
            calibrated_model=calibrated_model,
            options=solver_prior_options,
        )
        progress.step(f"solving {held_out.subject_name}")
        assignment = _solve_loso_assignment_with_priors(
            held_out,
            config=config,
            calibrated_model=calibrated_model,
            parameters=tuning_result.parameters,
        )
        predicted_matrix = tracks_to_suite2p_index_matrix(
            assignment.result.tracks, held_out.sessions
        )
        base_scores = _score_prediction_against_reference(
            predicted_matrix, held_out.reference, config=config
        )
        calibration_scores = _score_holdout_calibration(
            calibrated_model,
            held_out,
            config=config,
            feature_names=feature_names,
        )
        scores: dict[str, float | int | str] = {
            **base_scores,
            **training_scores,
            **tuning_result.to_score_dict(),
            **calibration_scores,
        }
        training_examples = int(training_scores["training_examples"])
        positive_examples = int(training_scores["positive_examples"])
        folds.append(
            LosoCalibrationFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(
                    subject.subject_name for subject in training_subjects
                ),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant=_variant_name_for_calibrated_solver_priors(calibration_model),
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.reference.n_sessions,
                    reference_source=held_out.reference.source,
                ),
                training_examples=training_examples,
                positive_examples=positive_examples,
            )
        )
    return LosoCalibrationResult(
        folds=tuple(folds), feature_names=feature_names, max_gap=int(config.max_gap)
    )


def tune_solver_priors_for_training_subjects(
    training_subjects: Sequence[SubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    cost: AssociationCost,
    calibrated_model: Any | None = None,
    options: SolverPriorTuningOptions | None = None,
) -> SolverPriorTuningResult:
    """Tune global-assignment priors on LOSO training subjects only."""

    if not training_subjects:
        raise ValueError(
            "At least one training subject is required for solver-prior tuning"
        )

    options = options or SolverPriorTuningOptions()
    candidates = _solver_prior_parameter_grid(config, options=options)
    cached_subjects = tuple(
        _cache_solver_tuning_subject(
            subject,
            config=config,
            cost=cost,
            calibrated_model=calibrated_model,
        )
        for subject in training_subjects
    )

    best: SolverPriorTuningResult | None = None
    for candidate in candidates:
        mean_scores = _score_solver_prior_candidate(
            candidate, cached_subjects, config=config
        )
        objective_value = float(mean_scores.get(options.objective, np.nan))
        if not np.isfinite(objective_value):
            continue
        if best is None or objective_value > best.objective_value:
            best = SolverPriorTuningResult(
                parameters=candidate,
                objective_name=options.objective,
                objective_value=objective_value,
                candidate_count=len(candidates),
                mean_scores=mean_scores,
            )
    if best is None:
        raise ValueError(
            f"No finite {options.objective!r} score was produced during solver-prior tuning"
        )
    return best


def _cache_solver_tuning_subject(
    subject: SubjectCalibrationData,
    *,
    config: Track2pBenchmarkConfig,
    cost: AssociationCost,
    calibrated_model: Any | None,
) -> CachedSolverTuningSubject:
    pairwise_costs = build_registered_pairwise_costs(
        subject.sessions,
        max_gap=config.max_gap,
        cost=cost,
        calibrated_model=calibrated_model,
        transform_type=config.transform_type,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
    )
    return CachedSolverTuningSubject(
        subject=subject,
        pairwise_costs=pairwise_costs,
        session_sizes=tuple(
            int(session.plane_data.n_rois) for session in subject.sessions
        ),
        session_edges=session_edge_pairs(len(subject.sessions), max_gap=config.max_gap),
    )


def _score_solver_prior_candidate(
    candidate: SolverPriorParameters,
    cached_subjects: Sequence[CachedSolverTuningSubject],
    *,
    config: Track2pBenchmarkConfig,
) -> dict[str, float]:
    score_rows: list[Mapping[str, float | int | str]] = []
    for cached in cached_subjects:
        assignment = solve_global_assignment_from_pairwise_costs(
            cached.pairwise_costs,
            session_sizes=cached.session_sizes,
            session_edges=cached.session_edges,
            start_cost=candidate.start_cost,
            end_cost=candidate.end_cost,
            gap_penalty=candidate.gap_penalty,
            cost_threshold=candidate.cost_threshold,
        )
        predicted_matrix = tracks_to_suite2p_index_matrix(
            assignment.result.tracks, cached.subject.sessions
        )
        score_rows.append(
            _score_prediction_against_reference(
                predicted_matrix,
                cached.subject.reference,
                config=config,
            )
        )
    return _mean_numeric_scores(score_rows)


def _solve_loso_assignment_with_priors(
    held_out: SubjectCalibrationData,
    *,
    config: Track2pBenchmarkConfig,
    calibrated_model: Any,
    parameters: SolverPriorParameters,
) -> Any:
    cached = _cache_solver_tuning_subject(
        held_out,
        config=config,
        cost="calibrated",
        calibrated_model=calibrated_model,
    )
    return solve_global_assignment_from_pairwise_costs(
        cached.pairwise_costs,
        session_sizes=cached.session_sizes,
        session_edges=cached.session_edges,
        start_cost=parameters.start_cost,
        end_cost=parameters.end_cost,
        gap_penalty=parameters.gap_penalty,
        cost_threshold=parameters.cost_threshold,
    )


# pylint: disable=too-many-arguments
def _fit_calibrated_cost_model(
    training_subjects: Sequence[SubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    calibration_model: CalibrationModelKind,
    sample_weight: Any | None,
    sample_weight_strategy: SampleWeightStrategy,
    logistic_model_kwargs: Mapping[str, Any],
    monotone_options: MonotoneRankerOptions,
    progress: ProgressReporter | None,
    held_out_subject: str,
) -> tuple[Any, dict[str, float | int | str]]:
    if calibration_model == "logistic":
        training_features, training_labels = _collect_training_examples(
            training_subjects,
            config=config,
            feature_names=feature_names,
            progress=progress,
            held_out_subject=held_out_subject,
        )
        weights = _training_sample_weight(
            training_labels,
            sample_weight=sample_weight,
            strategy=sample_weight_strategy,
        )
        if progress is not None:
            progress.step(f"fitting logistic model for {held_out_subject}")
        calibrated_model = fit_logistic_association_model(
            training_features,
            training_labels,
            feature_names=feature_names,
            sample_weight=weights,
            model_kwargs=logistic_model_kwargs,
        )
        positives = int(np.sum(training_labels))
        return calibrated_model, {
            "training_examples": int(training_labels.shape[0]),
            "positive_examples": positives,
            "negative_examples": int(training_labels.shape[0] - positives),
            "calibration_model": "logistic",
            "calibration_feature_names": ",".join(feature_names),
            "calibration_sample_weight_strategy": sample_weight_strategy,
            "calibration_class_weight": _stringify_class_weight(
                logistic_model_kwargs.get("class_weight")
            ),
        }

    if calibration_model == "monotone":
        if sample_weight is not None or sample_weight_strategy != "none":
            raise ValueError(
                "sample weights are only supported for calibration_model='logistic'"
            )
        blocks = _collect_training_blocks(
            training_subjects,
            config=config,
            feature_names=feature_names,
            progress=progress,
            held_out_subject=held_out_subject,
        )
        if progress is not None:
            progress.step(f"fitting monotone ranker for {held_out_subject}")
        calibrated_model = fit_monotone_ranking_association_model_from_blocks(
            blocks,
            options=monotone_options,
        )
        positives = int(calibrated_model.n_positive_examples)
        training_examples = int(calibrated_model.n_training_examples)
        return calibrated_model, {
            "training_examples": training_examples,
            "positive_examples": positives,
            "negative_examples": int(training_examples - positives),
            "calibration_model": "monotone-ranker",
            "calibration_feature_names": ",".join(feature_names),
            "monotone_feature_names": ",".join(
                calibrated_model.monotone_feature_names
            ),
            "monotone_rank_constraints": int(calibrated_model.n_rank_constraints),
            "monotone_training_rank_loss": float(
                calibrated_model.training_rank_loss
            ),
            "monotone_training_binary_loss": float(
                calibrated_model.training_binary_loss
            ),
            **_monotone_option_scores(monotone_options),
        }

    raise ValueError("calibration_model must be either 'logistic' or 'monotone'")


def _collect_training_blocks(
    training_subjects: Sequence[SubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    progress: ProgressReporter | None,
    held_out_subject: str,
) -> tuple[Any, ...]:
    blocks: list[Any] = []
    training_options = _reference_training_options(config, feature_names)
    for subject in training_subjects:
        if progress is not None:
            progress.step(
                f"collecting {subject.subject_name} ranking blocks for {held_out_subject}"
            )
        blocks.extend(
            collect_reference_pairwise_example_blocks(
                subject.sessions,
                subject.reference,
                session_edges=session_edge_pairs(
                    len(subject.sessions), max_gap=config.max_gap
                ),
                options=training_options,
            )
        )
    if not blocks:
        raise ValueError("At least one training block is required")
    return tuple(blocks)


def _score_holdout_calibration(
    calibrated_model: Any,
    held_out: SubjectCalibrationData,
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
) -> dict[str, float | int]:
    features, labels = collect_reference_training_examples(
        held_out.sessions,
        held_out.reference,
        session_edges=session_edge_pairs(len(held_out.sessions), max_gap=config.max_gap),
        options=_reference_training_options(config, feature_names),
    )
    probabilities = np.asarray(
        _predict_match_probability(calibrated_model, features), dtype=float
    ).reshape(-1)
    return calibration_summary(probabilities, np.asarray(labels).reshape(-1))


def _predict_match_probability(calibrated_model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(calibrated_model, "predict_match_probability"):
        return np.asarray(calibrated_model.predict_match_probability(features))
    return np.asarray(calibrated_model.model.predict_match_probability(features))


def _monotone_option_scores(
    options: MonotoneRankerOptions,
) -> dict[str, float | int | str]:
    values = asdict(options)
    return {
        f"monotone_option_{key}": ",".join(value) if isinstance(value, tuple) else value
        for key, value in values.items()
    }


def _variant_name_for_calibrated_solver_priors(
    calibration_model: CalibrationModelKind,
) -> str:
    if calibration_model == "monotone":
        return "Monotone ranker costs + LOSO tuned-prior global assignment"
    return "Calibrated costs + LOSO tuned-prior global assignment"


def _mean_numeric_scores(
    rows: Sequence[Mapping[str, float | int | str]],
) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_value = float(value)
                if np.isfinite(numeric_value):
                    values_by_key.setdefault(key, []).append(numeric_value)
    return {
        key: float(np.mean(values)) for key, values in values_by_key.items() if values
    }


def _solver_prior_parameter_grid(
    config: Track2pBenchmarkConfig,
    *,
    options: SolverPriorTuningOptions,
) -> tuple[SolverPriorParameters, ...]:
    starts = _solver_float_grid(
        options.start_costs,
        defaults=DEFAULT_SOLVER_PRIOR_START_COSTS,
        current=config.start_cost,
        positive=True,
        name="solver prior start costs",
    )
    ends = _solver_float_grid(
        options.end_costs,
        defaults=DEFAULT_SOLVER_PRIOR_END_COSTS,
        current=config.end_cost,
        positive=True,
        name="solver prior end costs",
    )
    gaps = _solver_float_grid(
        options.gap_penalties,
        defaults=DEFAULT_SOLVER_PRIOR_GAP_PENALTIES,
        current=config.gap_penalty,
        positive=False,
        name="solver prior gap penalties",
    )
    thresholds = _solver_threshold_grid(
        options.cost_thresholds,
        defaults=DEFAULT_SOLVER_PRIOR_COST_THRESHOLDS,
        current=config.cost_threshold,
    )
    return tuple(
        SolverPriorParameters(start, end, gap_penalty, threshold)
        for start in starts
        for end in ends
        for gap_penalty in gaps
        for threshold in thresholds
    )


def _solver_float_grid(
    configured: Sequence[float] | None,
    *,
    defaults: Sequence[float],
    current: float,
    positive: bool,
    name: str,
) -> tuple[float, ...]:
    values = (
        tuple(configured) if configured is not None else (*defaults, float(current))
    )
    values = _dedupe_float_values(values)
    if not values:
        raise ValueError(f"At least one {name} value is required")
    for value in values:
        invalid = value <= 0.0 if positive else value < 0.0
        if (not np.isfinite(value)) or invalid:
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} values must be {qualifier} finite numbers")
    return values


def _solver_threshold_grid(
    configured: Sequence[float | None] | None,
    *,
    defaults: Sequence[float | None],
    current: float | None,
) -> tuple[float | None, ...]:
    current_value = None if current is None else float(current)
    values = tuple(configured) if configured is not None else (*defaults, current_value)
    values = _dedupe_threshold_values(values)
    if not values:
        raise ValueError("At least one solver prior cost threshold is required")
    for value in values:
        if value is not None and ((not np.isfinite(value)) or value < 0.0):
            raise ValueError(
                "solver prior cost thresholds must be non-negative finite numbers or none"
            )
    return values


def _dedupe_float_values(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        numeric_value = float(value)
        if numeric_value not in result:
            result.append(numeric_value)
    return tuple(result)


def _dedupe_threshold_values(
    values: Sequence[float | None],
) -> tuple[float | None, ...]:
    result: list[float | None] = []
    for value in values:
        numeric_value = None if value is None else float(value)
        if numeric_value not in result:
            result.append(numeric_value)
    return tuple(result)


def _threshold_label(threshold: float | None) -> float | str:
    return "none" if threshold is None else float(threshold)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the calibrated/monotone solver-prior tuning parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-solver-prior-loso --cost calibrated",
        description=(
            "Run Track2p LOSO calibrated or monotone global assignment while "
            "tuning solver priors inside each training fold."
        ),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cost", default="calibrated", choices=("calibrated",))
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=(
            "affine",
            "rigid",
            "fov-translation",
            "fov-affine",
            "bspline",
            "tps",
            "thin-plate-spline",
            "local-affine-grid",
            "optical-flow",
            "none",
        ),
    )
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument("--no-cost-threshold", action="store_true")
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
    parser.add_argument(
        "--calibration-model",
        default="logistic",
        choices=("logistic", "monotone"),
        help="Learned edge-cost model to fit inside each LOSO training fold.",
    )
    parser.add_argument("--feature-names", default=",".join(DEFAULT_ASSOCIATION_FEATURES))
    parser.add_argument(
        "--sample-weight-strategy",
        default="none",
        choices=("none", "balanced"),
        help="Logistic calibration sample weighting; monotone ranker requires none.",
    )
    parser.add_argument("--model-kwargs-json", default=None)
    parser.add_argument("--monotone-ranker-kwargs-json", default=None)
    parser.add_argument(
        "--objective",
        default="pairwise_f1",
        choices=("pairwise_f1", "complete_track_f1"),
    )
    parser.add_argument("--start-costs", default=None)
    parser.add_argument("--end-costs", default=None)
    parser.add_argument("--gap-penalties", default=None)
    parser.add_argument("--cost-thresholds", default=None)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    options = SolverPriorTuningOptions(
        objective=cast(SolverPriorObjective, args.objective),
        start_costs=_parse_optional_float_tuple(args.start_costs, name="--start-costs"),
        end_costs=_parse_optional_float_tuple(args.end_costs, name="--end-costs"),
        gap_penalties=_parse_optional_float_tuple(args.gap_penalties, name="--gap-penalties"),
        cost_thresholds=_parse_optional_threshold_tuple(args.cost_thresholds),
    )
    rows = run_track2p_loso_solver_prior_tuning(
        config,
        feature_names=_parse_feature_names(args.feature_names),
        sample_weight_strategy=cast(SampleWeightStrategy, args.sample_weight_strategy),
        model_kwargs=_parse_json_object(args.model_kwargs_json, name="--model-kwargs-json"),
        calibration_model=cast(CalibrationModelKind, args.calibration_model),
        monotone_options=_monotone_options_from_json(args.monotone_ranker_kwargs_json),
        solver_prior_options=options,
    ).to_rows()
    if args.output is not None:
        from bayescatrack.experiments.track2p_benchmark import write_results

        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        _write_stdout(rows, cast(OutputFormat, args.format))
    return 0


def _config_from_args(args: argparse.Namespace) -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="leave-one-subject-out",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        cost="calibrated",
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
        pairwise_cost_kwargs=_parse_json_object(
            args.pairwise_cost_kwargs_json, name="--pairwise-cost-kwargs-json"
        ),
        progress=args.progress,
    )


def _parse_feature_names(raw: str) -> tuple[str, ...]:
    names = tuple(token.strip() for token in raw.split(","))
    if not names or any(not name for name in names):
        raise ValueError("--feature-names must be a comma-separated list without empty entries")
    return names


def _parse_json_object(raw: str | None, *, name: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return parsed


def _monotone_options_from_json(raw: str | None) -> MonotoneRankerOptions:
    parsed = _parse_json_object(raw, name="--monotone-ranker-kwargs-json")
    if parsed is None:
        return MonotoneRankerOptions()
    return MonotoneRankerOptions(**parsed)


def _parse_optional_float_tuple(raw: str | None, *, name: str) -> tuple[float, ...] | None:
    if raw is None:
        return None
    return tuple(_parse_float_token(token, name=name) for token in _split_csv(raw, name=name))


def _parse_optional_threshold_tuple(raw: str | None) -> tuple[float | None, ...] | None:
    if raw is None:
        return None
    values: list[float | None] = []
    for token in _split_csv(raw, name="--cost-thresholds"):
        values.append(
            None
            if token.lower() in {"none", "null", "off", "disabled"}
            else _parse_float_token(token, name="--cost-thresholds")
        )
    return tuple(values)


def _split_csv(raw: str, *, name: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(","))
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{name} must be a comma-separated list without empty entries")
    return tokens


def _parse_float_token(token: str, *, name: str) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"{name} contains a non-numeric value: {token!r}") from exc


def _write_stdout(
    rows: Sequence[dict[str, float | int | str]], output_format: OutputFormat
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_benchmark_table(rows))


def _csv_fieldnames(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "n_sessions",
        "reference_source",
        "pairwise_f1",
        "complete_track_f1",
        "training_examples",
        "positive_examples",
        "negative_examples",
        "calibration_model",
        "solver_prior_objective",
        "tuned_start_cost",
        "tuned_end_cost",
        "tuned_gap_penalty",
        "tuned_cost_threshold",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
