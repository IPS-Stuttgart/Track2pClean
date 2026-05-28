"""Gap-balanced leave-one-subject-out calibration for Track2p benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayescatrack.association.pyrecest_global_assignment import (
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.experiments.calibration_gap_weights import (
    balanced_binary_gap_sample_weights,
)
from bayescatrack.experiments.calibration_hard_negatives import (
    CandidateHardNegativeOptions,
)
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _maybe_refine_predicted_tracks,
    _score_prediction_against_reference,
    format_benchmark_table,
    solve_configured_global_assignment,
    write_results,
)
from bayescatrack.experiments.track2p_configurable_loso_calibration import (
    CalibrationModelKind,
    _class_weight_label,
    _collect_examples,
    _config_from_args,
    _fit_model,
    _hard_negative_options,
    _hard_negative_scores,
    _json_object,
    _load_subjects,
    _model_kwargs,
    _resolved_calibration_feature_names,
    _resolved_feature_names,
    _validate_calibration_model_kind,
    build_arg_parser as _configurable_arg_parser,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    LosoCalibrationFold,
    LosoCalibrationResult,
    config_with_fold_learned_gap_priors,
)


# pylint: disable=too-many-arguments,too-many-locals
def run_track2p_gap_balanced_loso_calibration(
    config: Track2pBenchmarkConfig,
    *,
    feature_names: Sequence[str] | None = None,
    model_kind: CalibrationModelKind = "logistic",
    model_kwargs: Mapping[str, Any] | None = None,
    hard_negative_options: CandidateHardNegativeOptions | None = None,
) -> LosoCalibrationResult:
    """Run LOSO calibrated assignment with label-and-session-gap sample weights.

    This is an opt-in variant for Track2p-style data sets where adjacent-session
    and skip-session association examples are pooled during calibration.  The
    sample weights equalize the total mass of each observed
    ``(match_label, session_gap)`` group, so rarer skip-edge positives and hard
    negatives can influence the calibrated association model rather than being
    dominated by adjacent-session candidates.
    """

    if config.method != "global-assignment" or config.cost != "calibrated":
        raise ValueError(
            "LOSO calibration requires method='global-assignment' and cost='calibrated'"
        )
    feature_names = _resolved_feature_names(config, feature_names)
    subjects = _load_subjects(config)
    model_kind = _validate_calibration_model_kind(model_kind)
    model_kwargs = _model_kwargs(model_kind, model_kwargs)
    hard_negative_options = hard_negative_options or CandidateHardNegativeOptions()
    progress = ProgressReporter(
        len(subjects) * (len(subjects) + 2),
        enabled=config.progress,
        label="gap-balanced LOSO",
    )
    folds: list[LosoCalibrationFold] = []
    for held_out_index, held_out in enumerate(subjects):
        training_subjects = tuple(
            subject for index, subject in enumerate(subjects) if index != held_out_index
        )
        features, labels = _collect_examples(
            training_subjects,
            config,
            feature_names,
            hard_negative_options,
            progress,
            held_out.subject_name,
        )
        weights = balanced_binary_gap_sample_weights(features, labels, feature_names)
        progress.step(f"fitting gap-balanced model for {held_out.subject_name}")
        model = _fit_model(
            features,
            labels,
            feature_names=feature_names,
            sample_weight=weights,
            model_kind=model_kind,
            model_kwargs=model_kwargs,
        )
        solve_config = config_with_fold_learned_gap_priors(config, training_subjects)
        progress.step(f"solving {held_out.subject_name}")
        assignment = solve_configured_global_assignment(
            held_out.sessions,
            solve_config,
            cost="calibrated",
            calibrated_model=model,
        )
        predicted = tracks_to_suite2p_index_matrix(
            assignment.result.tracks,
            held_out.sessions,
        )
        predicted = _maybe_refine_predicted_tracks(
            predicted,
            held_out.sessions,
            config=solve_config,
        )
        positives = int(np.sum(labels))
        scores: dict[str, float | int | str] = {
            **_score_prediction_against_reference(
                predicted,
                held_out.reference,
                config=config,
            ),
            "training_examples": int(labels.shape[0]),
            "positive_examples": positives,
            "negative_examples": int(labels.shape[0] - positives),
            "calibration_model": model_kind,
            "calibration_feature_count": int(len(feature_names)),
            "calibration_model_kwargs": json.dumps(
                dict(model_kwargs),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "calibration_sample_weight_strategy": "gap-balanced",
            "calibration_class_weight": _class_weight_label(model_kwargs),
            "calibration_feature_names": ",".join(feature_names),
            "gap_balanced_weight_min": float(np.min(weights)),
            "gap_balanced_weight_max": float(np.max(weights)),
            "gap_balanced_weight_sum": float(np.sum(weights)),
            **_hard_negative_scores(hard_negative_options),
        }
        folds.append(
            LosoCalibrationFold(
                held_out_subject=held_out.subject_name,
                training_subjects=tuple(
                    subject.subject_name for subject in training_subjects
                ),
                benchmark=SubjectBenchmarkResult(
                    subject=held_out.subject_name,
                    variant="Gap-balanced calibrated costs + LOSO global assignment",
                    method=config.method,
                    scores=scores,
                    n_sessions=held_out.reference.n_sessions,
                    reference_source=held_out.reference.source,
                ),
                training_examples=int(labels.shape[0]),
                positive_examples=positives,
            )
        )
    return LosoCalibrationResult(
        folds=tuple(folds), feature_names=tuple(feature_names), max_gap=int(config.max_gap)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Return CLI parser for the gap-balanced calibrated LOSO variant."""

    parser = _configurable_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-gap-balanced-loso-calibration"
    parser.description = (
        "Run configurable Track2p LOSO calibration with gap-balanced sample weights."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    result = run_track2p_gap_balanced_loso_calibration(
        config,
        feature_names=_resolved_calibration_feature_names(args),
        model_kind=args.calibration_model,
        model_kwargs=_json_object(
            args.calibration_model_kwargs_json,
            "--calibration-model-kwargs-json",
        ),
        hard_negative_options=_hard_negative_options(args),
    )
    rows = result.to_rows()
    if args.output is None:
        _write_stdout(rows, args.format)
    else:
        write_results(rows, args.output, args.format)
    return 0


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
        "pairwise_f1",
        "complete_track_f1",
        "training_examples",
        "positive_examples",
        "negative_examples",
        "calibration_model",
        "calibration_sample_weight_strategy",
        "gap_balanced_weight_min",
        "gap_balanced_weight_max",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
