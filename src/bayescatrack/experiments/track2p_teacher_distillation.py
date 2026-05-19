"""Track2p-teacher-distilled monotone ranker benchmark."""

# jscpd:ignore-start
# pylint: disable=protected-access
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    ReferenceTrainingOptions,
    collect_reference_pairwise_example_blocks,
    collect_reference_training_examples,
)
from bayescatrack.association.monotone_ranker import (
    MonotoneRankerOptions,
    fit_monotone_ranking_association_model_from_blocks,
)
from bayescatrack.association.pyrecest_global_assignment import (
    session_edge_pairs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.evaluation.calibration_diagnostics import calibration_summary
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _score_prediction_against_reference,
    discover_subject_dirs,
    solve_configured_global_assignment,
    write_results,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    LosoCalibrationFold,
    LosoCalibrationResult,
    SubjectCalibrationData,
    _load_subject_calibration_data,
    _reference_training_options,
)
from bayescatrack.experiments.track2p_monotone_loso_calibration import (
    _monotone_option_scores,
    _monotone_options_from_args,
    _write_stdout,
)
from bayescatrack.reference import Track2pReference


@dataclass(frozen=True)
class TeacherSubjectCalibrationData:
    """Loaded sessions plus manual-GT and Track2p-teacher identities."""

    subject_dir: Path
    sessions: tuple[Any, ...]
    manual_reference: Track2pReference
    teacher_reference: Track2pReference

    @property
    def subject_name(self) -> str:
        return self.subject_dir.name


# pylint: disable=too-many-arguments,too-many-locals
def run_track2p_teacher_distilled_loso(
    config: Track2pBenchmarkConfig,
    *,
    teacher_reference: Path | None = None,
    teacher_curated_only: bool | None = None,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    monotone_options: MonotoneRankerOptions | None = None,
) -> LosoCalibrationResult:
    """Run LOSO assignment using a monotone ranker trained on Track2p labels.

    The held-out subject is never used for fitting.  Manual GT is used only for
    held-out scoring/calibration diagnostics; Track2p output acts as a teacher
    reference for the training folds and as an optional agreement diagnostic on
    the held-out fold.
    """

    _validate_teacher_distillation_config(config)
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("Teacher-distilled LOSO requires at least two subjects")

    progress = ProgressReporter(
        len(subject_dirs) + len(subject_dirs) * (len(subject_dirs) + 3),
        enabled=config.progress,
        label="teacher-distill-loso",
    )
    subjects: list[TeacherSubjectCalibrationData] = []
    for subject_dir in subject_dirs:
        progress.step(f"loading {subject_dir.name}")
        subjects.append(
            _load_teacher_subject(
                subject_dir,
                config=config,
                teacher_reference=teacher_reference,
            )
        )

    options = monotone_options or MonotoneRankerOptions()
    feature_names = tuple(feature_names)
    folds: list[LosoCalibrationFold] = []
    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(
            subject for index, subject in enumerate(subjects) if index != held_out_index
        )
        progress.step(f"collecting Track2p-teacher blocks for {held_out.subject_name}")
        training_blocks = _collect_teacher_training_blocks(
            training_subjects,
            config=config,
            feature_names=feature_names,
            teacher_curated_only=teacher_curated_only,
        )
        progress.step(f"fitting teacher-distilled ranker for {held_out.subject_name}")
        calibrated_model = fit_monotone_ranking_association_model_from_blocks(
            training_blocks,
            options=options,
        )
        progress.step(f"scoring manual-GT calibration for {held_out.subject_name}")
        manual_calibration_scores = _score_holdout_calibration_against_reference(
            calibrated_model,
            held_out,
            reference=held_out.manual_reference,
            config=config,
            feature_names=feature_names,
            prefix="manual_gt",
        )
        progress.step(f"scoring teacher agreement for {held_out.subject_name}")
        teacher_calibration_scores = _score_holdout_calibration_against_reference(
            calibrated_model,
            held_out,
            reference=held_out.teacher_reference,
            config=config,
            feature_names=feature_names,
            prefix="teacher",
            curated_only=teacher_curated_only,
        )
        progress.step(f"solving {held_out.subject_name}")
        assignment = solve_configured_global_assignment(
            held_out.sessions,
            config,
            cost="calibrated",
            calibrated_model=calibrated_model,
        )
        predicted_matrix = tracks_to_suite2p_index_matrix(
            assignment.result.tracks, held_out.sessions
        )
        base_scores = _score_prediction_against_reference(
            predicted_matrix, held_out.manual_reference, config=config
        )
        positives = int(calibrated_model.n_positive_examples)
        training_examples = int(calibrated_model.n_training_examples)
        scores: dict[str, float | int | str] = {
            **base_scores,
            "training_examples": training_examples,
            "positive_examples": positives,
            "negative_examples": int(training_examples - positives),
            "calibration_model": "track2p-teacher-distilled-monotone-ranker",
            "teacher_label_source": "track2p_output",
            "teacher_training_subjects": ",".join(
                subject.subject_name for subject in training_subjects
            ),
            "teacher_reference_source": held_out.teacher_reference.source,
            "monotone_feature_names": ",".join(calibrated_model.monotone_feature_names),
            "monotone_rank_constraints": int(calibrated_model.n_rank_constraints),
            "monotone_training_rank_loss": float(calibrated_model.training_rank_loss),
            "monotone_training_binary_loss": float(
                calibrated_model.training_binary_loss
            ),
            **_monotone_option_scores(options),
            **manual_calibration_scores,
            **teacher_calibration_scores,
        }
        folds.append(
            LosoCalibrationFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(
                    subject.subject_name for subject in training_subjects
                ),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant=(
                        "Track2p teacher-distilled monotone ranker + "
                        "LOSO global assignment"
                    ),
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.manual_reference.n_sessions,
                    reference_source=held_out.manual_reference.source,
                ),
                training_examples=training_examples,
                positive_examples=positives,
            )
        )
    return LosoCalibrationResult(
        folds=tuple(folds), feature_names=feature_names, max_gap=int(config.max_gap)
    )


def _validate_teacher_distillation_config(config: Track2pBenchmarkConfig) -> None:
    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "Teacher distillation requires method='global-assignment' and cost='calibrated'"
        )
    if config.split != "leave-one-subject-out":
        raise ValueError("Teacher distillation requires split='leave-one-subject-out'")
    if config.reference_kind != "manual-gt":
        raise ValueError(
            "Teacher distillation evaluates against independent manual GT; "
            "use --reference-kind manual-gt"
        )
    if config.allow_track2p_as_reference_for_smoke_test:
        raise ValueError(
            "Teacher distillation deliberately separates Track2p teacher labels "
            "from manual-GT evaluation; do not pass "
            "--allow-track2p-as-reference-for-smoke-test"
        )


def _load_teacher_subject(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    teacher_reference: Path | None,
) -> TeacherSubjectCalibrationData:
    manual_subject: SubjectCalibrationData = _load_subject_calibration_data(
        subject_dir, config=config
    )
    if manual_subject.reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError(
            f"Subject {subject_dir.name!r} did not resolve independent manual GT"
        )
    teacher_config = replace(
        config,
        reference=teacher_reference,
        reference_kind="track2p-output",
        allow_track2p_as_reference_for_smoke_test=True,
    )
    teacher = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=teacher_config
    )
    if teacher.n_sessions != len(manual_subject.sessions):
        raise ValueError(
            f"Subject {subject_dir.name!r} has {len(manual_subject.sessions)} loaded "
            f"sessions but {teacher.n_sessions} Track2p-teacher sessions"
        )
    return TeacherSubjectCalibrationData(
        subject_dir=subject_dir,
        sessions=manual_subject.sessions,
        manual_reference=manual_subject.reference,
        teacher_reference=teacher,
    )


def _collect_teacher_training_blocks(
    training_subjects: Sequence[TeacherSubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    teacher_curated_only: bool | None,
) -> tuple[Any, ...]:
    blocks: list[Any] = []
    for subject in training_subjects:
        options = _teacher_training_options(
            config,
            feature_names,
            subject.teacher_reference,
            teacher_curated_only=teacher_curated_only,
        )
        blocks.extend(
            collect_reference_pairwise_example_blocks(
                subject.sessions,
                subject.teacher_reference,
                session_edges=session_edge_pairs(
                    len(subject.sessions), max_gap=config.max_gap
                ),
                options=options,
            )
        )
    if not blocks:
        raise ValueError("At least one teacher-training block is required")
    return tuple(blocks)


def _teacher_training_options(
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    teacher_reference: Track2pReference,
    *,
    teacher_curated_only: bool | None,
) -> ReferenceTrainingOptions:
    requested_curated = (
        config.curated_only if teacher_curated_only is None else teacher_curated_only
    )
    curated = bool(requested_curated and teacher_reference.curated_mask is not None)
    return replace(
        _reference_training_options(config, feature_names),
        curated_only=curated,
    )


def _score_holdout_calibration_against_reference(
    calibrated_model: Any,
    held_out: TeacherSubjectCalibrationData,
    *,
    reference: Track2pReference,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
    prefix: str,
    curated_only: bool | None = None,
) -> dict[str, float | int]:
    options = _reference_training_options(config, feature_names)
    requested_curated = options.curated_only if curated_only is None else curated_only
    options = replace(
        options,
        curated_only=bool(requested_curated and reference.curated_mask is not None),
    )
    features, labels = collect_reference_training_examples(
        held_out.sessions,
        reference,
        session_edges=session_edge_pairs(
            len(held_out.sessions), max_gap=config.max_gap
        ),
        options=options,
    )
    probabilities = np.asarray(
        calibrated_model.predict_match_probability(features), dtype=float
    ).reshape(-1)
    scores = calibration_summary(probabilities, np.asarray(labels).reshape(-1))
    return {f"{prefix}_{key}": value for key, value in scores.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the teacher-distilled LOSO benchmark parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-teacher-distill",
        description=(
            "Run LOSO global assignment with a monotone ranker trained on "
            "Track2p teacher edges and evaluated against manual GT."
        ),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--teacher-reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("manual-gt",),
    )
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument(
        "--teacher-curated-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use Track2p's curated mask for teacher positives when available. "
            "Defaults to --curated-only."
        ),
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument("--monotone-ranker-kwargs-json", default=None)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    options = _monotone_options_from_args(args)
    rows = [
        fold.benchmark.to_dict()
        for fold in run_track2p_teacher_distilled_loso(
            config,
            teacher_reference=args.teacher_reference,
            teacher_curated_only=args.teacher_curated_only,
            monotone_options=options,
        ).folds
    ]
    if args.output is not None:
        write_results(rows, args.output, args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


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
        split="leave-one-subject-out",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=False,
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
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        progress=args.progress,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
# jscpd:ignore-end
