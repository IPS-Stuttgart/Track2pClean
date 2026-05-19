"""Track2p-teacher distillation for calibrated LOSO benchmarks.

This experiment trains the calibrated pairwise association model with optional
Track2p-output pseudo-labels while keeping the held-out benchmark score against
independent manual ground truth.  It is intentionally separate from the default
``track2p`` benchmark command so teacher supervision cannot accidentally leak
into paper-facing baseline runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    ReferencePairwiseExamples,
    ReferenceTrainingOptions,
    collect_reference_pairwise_example_blocks,
    fit_logistic_association_model,
)
from bayescatrack.association.pyrecest_global_assignment import (
    session_edge_pairs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.evaluation.calibration_diagnostics import calibration_summary
from bayescatrack.experiments.calibration_hard_negatives import (
    CandidateHardNegativeOptions,
    collect_candidate_limited_training_examples,
)
# pylint: disable=protected-access
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _config_from_args,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    solve_configured_global_assignment,
    write_results,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    LosoCalibrationFold,
    LosoCalibrationResult,
    SampleWeightStrategy,
    _loso_logistic_model_kwargs,
    _reference_training_options,
    _training_sample_weight,
    _validate_sample_weight_strategy,
)
from bayescatrack.reference import Track2pReference


@dataclass(frozen=True)
class TeacherDistillationOptions:
    """Controls for Track2p-teacher pseudo-label training."""

    teacher_reference: Path | None = None
    include_manual_training_labels: bool = True
    include_teacher_training_labels: bool = True
    manual_label_weight: float = 1.0
    teacher_label_weight: float = 0.5


@dataclass(frozen=True)
class TeacherSubjectCalibrationData:
    """Loaded sessions plus manual and Track2p-teacher references."""

    subject_dir: Path
    sessions: tuple[Track2pSession, ...]
    reference: Track2pReference
    teacher_reference: Track2pReference

    @property
    def subject_name(self) -> str:
        return self.subject_dir.name


@dataclass(frozen=True)
class WeightedTrainingExamples:
    """Candidate-limited examples with source weights."""

    features: np.ndarray
    labels: np.ndarray
    source_weights: np.ndarray | None


# pylint: disable=too-many-arguments,too-many-locals
def run_track2p_teacher_distillation(
    config: Track2pBenchmarkConfig,
    *,
    distillation: TeacherDistillationOptions | None = None,
    feature_names: Sequence[str] = DEFAULT_ASSOCIATION_FEATURES,
    sample_weight: Any | None = None,
    sample_weight_strategy: SampleWeightStrategy = "none",
    model_kwargs: Mapping[str, Any] | None = None,
) -> LosoCalibrationResult:
    """Run LOSO calibrated assignment with Track2p teacher pseudo-labels.

    The held-out subject is always scored against ``config``'s manual reference.
    Track2p output is used only as an additional source of candidate-edge labels
    on the non-held-out training subjects.  Set
    ``include_manual_training_labels=False`` to run a pure teacher-distillation
    ablation.
    """

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "Track2p-teacher distillation requires method='global-assignment' and cost='calibrated'"
        )
    if config.split != "leave-one-subject-out":
        raise ValueError(
            "Track2p-teacher distillation requires split='leave-one-subject-out'"
        )
    distillation = validate_teacher_distillation_options(
        distillation or TeacherDistillationOptions()
    )
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("Track2p-teacher distillation requires at least two subjects")

    progress = ProgressReporter(
        len(subject_dirs) + len(subject_dirs) * (len(subject_dirs) + 2),
        enabled=config.progress,
        label="teacher-distill",
    )
    subjects: list[TeacherSubjectCalibrationData] = []
    for subject_dir in subject_dirs:
        progress.step(f"loading {subject_dir.name}")
        subjects.append(
            _load_subject_teacher_data(
                subject_dir, config=config, distillation=distillation
            )
        )

    feature_names = tuple(feature_names)
    sample_weight_strategy = _validate_sample_weight_strategy(sample_weight_strategy)
    logistic_model_kwargs = _loso_logistic_model_kwargs(model_kwargs)
    folds: list[LosoCalibrationFold] = []

    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(
            subject for index, subject in enumerate(subjects) if index != held_out_index
        )
        examples = _collect_distilled_training_examples(
            training_subjects,
            config=config,
            distillation=distillation,
            feature_names=feature_names,
            progress=progress,
            held_out_subject=held_out.subject_name,
        )
        weights = _training_sample_weight(
            examples.labels,
            sample_weight=sample_weight,
            strategy=sample_weight_strategy,
        )
        weights = combine_sample_weights(weights, examples.source_weights)
        progress.step(f"fitting model for {held_out.subject_name}")
        calibrated_model = fit_logistic_association_model(
            examples.features,
            examples.labels,
            feature_names=feature_names,
            sample_weight=weights,
            model_kwargs=logistic_model_kwargs,
        )
        progress.step(f"scoring calibration for {held_out.subject_name}")
        calibration_scores = _score_holdout_calibration(
            calibrated_model,
            held_out,
            config=config,
            feature_names=feature_names,
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
            predicted_matrix, held_out.reference, config=config
        )
        positives = int(np.sum(examples.labels))
        scores: dict[str, float | int | str] = {
            **base_scores,
            "training_examples": int(examples.labels.shape[0]),
            "positive_examples": positives,
            "negative_examples": int(examples.labels.shape[0] - positives),
            "calibration_sample_weight_strategy": sample_weight_strategy,
            "calibration_class_weight": _stringify_class_weight(
                logistic_model_kwargs.get("class_weight")
            ),
            "distillation_include_manual_training_labels": int(
                distillation.include_manual_training_labels
            ),
            "distillation_include_teacher_training_labels": int(
                distillation.include_teacher_training_labels
            ),
            "distillation_manual_label_weight": float(distillation.manual_label_weight),
            "distillation_teacher_label_weight": float(distillation.teacher_label_weight),
            "distillation_teacher_reference_sources": ",".join(
                sorted({subject.teacher_reference.source for subject in training_subjects})
            ),
            **calibration_scores,
        }
        folds.append(
            LosoCalibrationFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(
                    subject.subject_name for subject in training_subjects
                ),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant=_variant_name(distillation),
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.reference.n_sessions,
                    reference_source=held_out.reference.source,
                ),
                training_examples=int(examples.labels.shape[0]),
                positive_examples=positives,
            )
        )
    return LosoCalibrationResult(
        folds=tuple(folds), feature_names=feature_names, max_gap=int(config.max_gap)
    )


def validate_teacher_distillation_options(
    options: TeacherDistillationOptions,
) -> TeacherDistillationOptions:
    """Validate and normalize teacher-distillation options."""

    if not options.include_manual_training_labels and not options.include_teacher_training_labels:
        raise ValueError(
            "At least one of manual or teacher training labels must be enabled"
        )
    if options.manual_label_weight <= 0.0:
        raise ValueError("manual_label_weight must be positive")
    if options.teacher_label_weight <= 0.0:
        raise ValueError("teacher_label_weight must be positive")
    return options


def combine_sample_weights(sample_weight: Any | None, source_weights: Any | None) -> Any | None:
    """Combine class/sample weights with manual-vs-teacher source weights."""

    if sample_weight is None and source_weights is None:
        return None
    if sample_weight is None:
        source = np.asarray(source_weights, dtype=float)
        return None if np.allclose(source, 1.0) else source
    if source_weights is None:
        return sample_weight

    base = np.asarray(sample_weight, dtype=float)
    source = np.asarray(source_weights, dtype=float)
    if base.shape == ():
        base = np.full(source.shape, float(base), dtype=float)
    if source.shape == ():
        source = np.full(base.shape, float(source), dtype=float)
    if base.shape != source.shape:
        raise ValueError(
            "sample_weight and source_weights must have the same shape or be scalar"
        )
    return base * source


def collect_weighted_candidate_examples(
    pairwise_blocks: Sequence[ReferencePairwiseExamples],
    *,
    source_weight: float,
    hard_negative_options: CandidateHardNegativeOptions | None = None,
) -> WeightedTrainingExamples:
    """Collect candidate-limited examples and attach a constant source weight."""

    if source_weight <= 0.0:
        raise ValueError("source_weight must be positive")
    features, labels = collect_candidate_limited_training_examples(
        pairwise_blocks, options=hard_negative_options or CandidateHardNegativeOptions()
    )
    weights = np.full(np.asarray(labels).shape, float(source_weight), dtype=float)
    return WeightedTrainingExamples(
        features=np.asarray(features, dtype=float),
        labels=np.asarray(labels, dtype=int),
        source_weights=weights,
    )


def _collect_distilled_training_examples(
    training_subjects: Sequence[TeacherSubjectCalibrationData],
    *,
    config: Track2pBenchmarkConfig,
    distillation: TeacherDistillationOptions,
    feature_names: Sequence[str],
    progress: ProgressReporter | None = None,
    held_out_subject: str | None = None,
) -> WeightedTrainingExamples:
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    training_options = _reference_training_options(config, feature_names)
    session_edges_cache: dict[int, tuple[tuple[int, int], ...]] = {}
    hard_negative_options = CandidateHardNegativeOptions()

    for subject in training_subjects:
        if progress is not None:
            progress.step(
                f"collecting {subject.subject_name} distilled features for {held_out_subject}"
            )
        session_edges = session_edges_cache.setdefault(
            len(subject.sessions),
            tuple(session_edge_pairs(len(subject.sessions), max_gap=config.max_gap)),
        )
        if distillation.include_manual_training_labels:
            manual_examples = _collect_one_reference_source(
                subject,
                reference=subject.reference,
                session_edges=session_edges,
                options=training_options,
                source_weight=distillation.manual_label_weight,
                hard_negative_options=hard_negative_options,
            )
            feature_blocks.append(manual_examples.features)
            label_blocks.append(manual_examples.labels)
            weight_blocks.append(cast(np.ndarray, manual_examples.source_weights))
        if distillation.include_teacher_training_labels:
            teacher_examples = _collect_one_reference_source(
                subject,
                reference=subject.teacher_reference,
                session_edges=session_edges,
                options=training_options,
                source_weight=distillation.teacher_label_weight,
                hard_negative_options=hard_negative_options,
            )
            feature_blocks.append(teacher_examples.features)
            label_blocks.append(teacher_examples.labels)
            weight_blocks.append(cast(np.ndarray, teacher_examples.source_weights))

    if not feature_blocks:
        raise ValueError("At least one training source is required")
    weights = np.concatenate(weight_blocks, axis=0) if weight_blocks else None
    return WeightedTrainingExamples(
        features=np.concatenate(feature_blocks, axis=0),
        labels=np.concatenate(label_blocks, axis=0),
        source_weights=weights,
    )


def _collect_one_reference_source(
    subject: TeacherSubjectCalibrationData,
    *,
    reference: Track2pReference,
    session_edges: Sequence[tuple[int, int]],
    options: ReferenceTrainingOptions,
    source_weight: float,
    hard_negative_options: CandidateHardNegativeOptions,
) -> WeightedTrainingExamples:
    pairwise_blocks = collect_reference_pairwise_example_blocks(
        subject.sessions,
        reference,
        session_edges=session_edges,
        options=options,
    )
    return collect_weighted_candidate_examples(
        pairwise_blocks,
        source_weight=source_weight,
        hard_negative_options=hard_negative_options,
    )


def _load_subject_teacher_data(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    distillation: TeacherDistillationOptions,
) -> TeacherSubjectCalibrationData:
    sessions = tuple(_load_subject_sessions(subject_dir, config))
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError("Track2p-teacher distillation requires manual-GT evaluation references")
    _validate_reference_roi_indices(reference, sessions)

    teacher_config = replace(
        config,
        reference=distillation.teacher_reference,
        reference_kind="track2p-output",
        allow_track2p_as_reference_for_smoke_test=True,
    )
    teacher_reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=teacher_config
    )
    _validate_teacher_reference_compatibility(
        teacher_reference, reference, subject_dir=subject_dir
    )
    _validate_reference_roi_indices(teacher_reference, sessions)
    return TeacherSubjectCalibrationData(
        subject_dir=subject_dir,
        sessions=sessions,
        reference=reference,
        teacher_reference=teacher_reference,
    )


def _validate_teacher_reference_compatibility(
    teacher_reference: Track2pReference,
    manual_reference: Track2pReference,
    *,
    subject_dir: Path,
) -> None:
    if teacher_reference.n_sessions != manual_reference.n_sessions:
        raise ValueError(
            f"Subject {subject_dir.name!r} has {teacher_reference.n_sessions} Track2p-teacher sessions but "
            f"{manual_reference.n_sessions} manual-GT sessions"
        )
    if teacher_reference.session_names != manual_reference.session_names:
        raise ValueError(
            f"Subject {subject_dir.name!r} Track2p-teacher session order "
            f"{teacher_reference.session_names!r} does not match manual-GT order "
            f"{manual_reference.session_names!r}"
        )


def _score_holdout_calibration(
    calibrated_model: Any,
    held_out: TeacherSubjectCalibrationData,
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
) -> dict[str, float | int]:
    blocks = collect_reference_pairwise_example_blocks(
        held_out.sessions,
        held_out.reference,
        session_edges=session_edge_pairs(len(held_out.sessions), max_gap=config.max_gap),
        options=_reference_training_options(config, feature_names),
    )
    feature_blocks = [block.features.reshape(-1, block.features.shape[-1]) for block in blocks]
    label_blocks = [block.labels.reshape(-1) for block in blocks]
    if not feature_blocks:
        raise ValueError("At least one hold-out edge is required")
    features = np.concatenate(feature_blocks, axis=0)
    labels = np.concatenate(label_blocks, axis=0)
    probabilities = np.asarray(
        calibrated_model.model.predict_match_probability(features), dtype=float
    ).reshape(-1)
    return calibration_summary(probabilities, labels.reshape(-1))


def _stringify_class_weight(class_weight: Any) -> str:
    return "None" if class_weight is None else str(class_weight)


def _variant_name(distillation: TeacherDistillationOptions) -> str:
    if distillation.include_manual_training_labels:
        return "Manual + Track2p-teacher distilled calibrated costs + LOSO global assignment"
    return "Track2p-teacher distilled calibrated costs + LOSO global assignment"


def build_arg_parser() -> argparse.ArgumentParser:
    from bayescatrack.experiments.track2p_benchmark import (  # pylint: disable=import-outside-toplevel
        build_arg_parser as _build_track2p_arg_parser,
    )

    parser = _build_track2p_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-teacher-distill"
    _set_action_default(parser, "method", "global-assignment", required=False)
    _set_action_default(parser, "split", "leave-one-subject-out")
    _set_action_default(parser, "cost", "calibrated")
    _set_action_default(parser, "reference_kind", "manual-gt")
    parser.add_argument(
        "--teacher-reference",
        type=Path,
        default=None,
        help="Optional Track2p-output root/folder used as pseudo-label teacher",
    )
    parser.add_argument(
        "--teacher-label-weight",
        type=float,
        default=0.5,
        help="Sample weight for Track2p-teacher pseudo-label examples",
    )
    parser.add_argument(
        "--manual-label-weight",
        type=float,
        default=1.0,
        help="Sample weight for manual-GT training examples from non-held-out subjects",
    )
    parser.add_argument(
        "--no-manual-training-labels",
        action="store_true",
        help="Train only on Track2p-teacher pseudo-labels in each LOSO fold",
    )
    parser.add_argument(
        "--no-teacher-training-labels",
        action="store_true",
        help="Disable teacher labels; useful for parity checks against ordinary LOSO",
    )
    parser.add_argument(
        "--sample-weight-strategy",
        choices=("none", "balanced"),
        default="none",
        help="Optional class reweighting multiplied with manual/teacher source weights",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    distillation = TeacherDistillationOptions(
        teacher_reference=args.teacher_reference,
        include_manual_training_labels=not args.no_manual_training_labels,
        include_teacher_training_labels=not args.no_teacher_training_labels,
        manual_label_weight=args.manual_label_weight,
        teacher_label_weight=args.teacher_label_weight,
    )
    results = run_track2p_teacher_distillation(
        config,
        distillation=distillation,
        sample_weight_strategy=cast(SampleWeightStrategy, args.sample_weight_strategy),
    )
    rows = results.to_rows()
    if args.output is not None:
        write_results(rows, args.output, args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


def _set_action_default(
    parser: argparse.ArgumentParser,
    dest: str,
    default: Any,
    *,
    required: bool | None = None,
) -> None:
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == dest:
            action.default = default
            if required is not None and hasattr(action, "required"):
                action.required = required
            return
    raise KeyError(f"Parser has no action named {dest!r}")


def _write_stdout(rows: Sequence[dict[str, float | int | str]], fmt: str) -> None:
    from bayescatrack.experiments.track2p_benchmark import (  # pylint: disable=import-outside-toplevel,protected-access
        _write_stdout as _track2p_write_stdout,
    )

    _track2p_write_stdout(rows, fmt)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
