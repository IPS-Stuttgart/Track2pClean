"""LOSO learned-score edge-ranking diagnostics for Track2p-style datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    ReferencePairwiseExamples,
    ReferenceTrainingOptions,
    collect_reference_pairwise_example_blocks,
    fit_logistic_association_model,
)
from bayescatrack.association.monotone_ranker import (
    MonotoneRankerOptions,
    fit_monotone_ranking_association_model_from_blocks,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    session_edge_pairs,
)
from bayescatrack.evaluation.edge_ranking import (
    DEFAULT_HIT_KS,
    ScoreDirection,
    missing_reference_edge_rows,
    rank_labeled_edges,
    summarize_edge_ranking_rows,
)
from bayescatrack.experiments.calibration_hard_negatives import (
    CandidateHardNegativeOptions,
    collect_candidate_limited_training_examples,
)
from bayescatrack.experiments.track2p_benchmark import (
    ProgressReporter,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_edge_ranking import (
    EDGE_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    _default_summary_output_path,
    _pairwise_cost_kwargs_for_config,
    _write_csv,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    pairwise_cost_kwargs_for_calibration_features,
)

LearnedScoreModel = Literal["logistic", "monotone"]
DEFAULT_LEARNED_EDGE_RANKING_FEATURES = DEFAULT_ASSOCIATION_FEATURES
_EPSILON = 1.0e-12


@dataclass(frozen=True)
class _LoadedSubjectBlocks:
    """Loaded subject data and materialized pairwise example blocks."""

    subject_dir: Path
    sessions: tuple[Any, ...]
    reference: Any
    blocks: tuple[ReferencePairwiseExamples, ...]

    @property
    def subject_name(self) -> str:
        return self.subject_dir.name


# pylint: disable=too-many-arguments,too-many-locals
def run_track2p_learned_edge_ranking(
    config: Track2pBenchmarkConfig,
    output_path: Path,
    *,
    summary_output_path: Path | None = None,
    score_model: LearnedScoreModel = "monotone",
    feature_names: Sequence[str] = DEFAULT_LEARNED_EDGE_RANKING_FEATURES,
    logistic_model_kwargs: Mapping[str, Any] | None = None,
    monotone_options: MonotoneRankerOptions | None = None,
    hit_ks: Sequence[int] = DEFAULT_HIT_KS,
) -> tuple[int, int]:
    """Export LOSO learned-score rankings for manual-GT ROI links.

    A separate learned model is fitted for each held-out subject from the other
    subjects' manual-GT pairwise blocks. The held-out blocks are then ranked by
    calibrated/logistic or monotone learned scores before any global-assignment
    solver is invoked. This directly diagnoses whether the learned pairwise
    score places each true edge above same-row and same-column alternatives.
    """

    score_model = _validate_score_model(score_model)
    feature_names = _unique_feature_names(feature_names)
    if not feature_names:
        raise ValueError("At least one learned-score feature is required")

    subject_dirs = tuple(discover_subject_dirs(config.data))
    if len(subject_dirs) < 2:
        raise ValueError("Learned-score edge-ranking requires at least two subjects")

    progress = ProgressReporter(
        len(subject_dirs) + len(subject_dirs),
        enabled=config.progress,
        label=f"{score_model}-edge-ranking",
    )
    loaded_subjects: list[_LoadedSubjectBlocks] = []
    for subject_dir in subject_dirs:
        progress.step(f"loading {subject_dir.name}")
        loaded_subjects.append(
            _load_subject_blocks(
                subject_dir,
                config=config,
                feature_names=feature_names,
            )
        )

    output_rows: list[dict[str, float | int | str]] = []
    for held_out_index, held_out in enumerate(loaded_subjects):
        training_subjects = tuple(
            subject
            for index, subject in enumerate(loaded_subjects)
            if index != held_out_index
        )
        progress.step(f"fitting {score_model} model for {held_out.subject_name}")
        learned_model = _fit_learned_score_model(
            score_model,
            _training_blocks(training_subjects),
            logistic_model_kwargs=logistic_model_kwargs,
            monotone_options=monotone_options,
        )
        training_subject_names = ",".join(
            subject.subject_name for subject in training_subjects
        )
        for block in held_out.blocks:
            score_matrices = _learned_score_matrices(
                learned_model,
                block,
                score_model=score_model,
            )
            score_directions = _score_directions_for_model(score_model)
            metadata = {
                "subject": held_out.subject_name,
                "session_a": int(block.session_a),
                "session_b": int(block.session_b),
                "session_a_name": held_out.sessions[block.session_a].session_name,
                "session_b_name": held_out.sessions[block.session_b].session_name,
                "session_gap": int(block.gap),
                "score_model": score_model,
                "training_subjects": training_subject_names,
                "model_feature_names": ",".join(feature_names),
            }
            output_rows.extend(
                rank_labeled_edges(
                    block.labels,
                    score_matrices,
                    reference_roi_indices=block.reference_roi_indices,
                    measurement_roi_indices=block.measurement_roi_indices,
                    score_directions=score_directions,
                    metadata=metadata,
                )
            )
            output_rows.extend(
                missing_reference_edge_rows(
                    held_out.reference.pairwise_matches(
                        block.session_a,
                        block.session_b,
                        curated_only=config.curated_only,
                    ),
                    reference_roi_indices=block.reference_roi_indices,
                    measurement_roi_indices=block.measurement_roi_indices,
                    score_names=tuple(score_matrices),
                    score_directions=score_directions,
                    metadata=metadata,
                )
            )

    output_path = Path(output_path)
    summary_path = (
        Path(summary_output_path)
        if summary_output_path is not None
        else _default_summary_output_path(output_path)
    )
    summary_rows = summarize_edge_ranking_rows(output_rows, hit_ks=hit_ks)
    _write_csv(output_rows, output_path, preferred_fieldnames=EDGE_FIELDNAMES)
    _write_csv(summary_rows, summary_path, preferred_fieldnames=SUMMARY_FIELDNAMES)
    return len(output_rows), len(summary_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for LOSO learned-score edge ranking."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-learned-edge-ranking",
        description=(
            "Export leave-one-subject-out calibrated/monotone learned-score "
            "edge-ranking diagnostics before global assignment."
        ),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--base-cost",
        dest="cost",
        default="roi-aware",
        choices=(
            "registered-iou",
            "registered-soft-iou",
            "registered-shifted-iou",
            "roi-aware",
            "roi-aware-shifted",
            "calibrated",
        ),
        help="Base component-generation cost used before fitting learned scores",
    )
    parser.add_argument(
        "--score-model",
        default="monotone",
        choices=("logistic", "monotone"),
        help=(
            "LOSO learned score to rank: logistic calibrated probabilities or "
            "monotone ranking costs"
        ),
    )
    parser.add_argument(
        "--feature",
        dest="features",
        action="append",
        default=None,
        help=(
            "Learned-score feature name; repeat to override the default "
            "association feature set"
        ),
    )
    parser.add_argument("--model-kwargs-json", default=None)
    parser.add_argument("--monotone-ranker-kwargs-json", default=None)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-translation", "none"),
    )
    parser.add_argument("--curated-only", action="store_true")
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
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    edge_rows, summary_rows = run_track2p_learned_edge_ranking(
        config,
        args.output,
        summary_output_path=args.summary_output,
        score_model=args.score_model,
        feature_names=(
            tuple(args.features)
            if args.features is not None
            else DEFAULT_LEARNED_EDGE_RANKING_FEATURES
        ),
        logistic_model_kwargs=_json_object_or_none(args.model_kwargs_json),
        monotone_options=_monotone_options_from_args(args),
    )
    summary_path = (
        args.summary_output
        if args.summary_output is not None
        else _default_summary_output_path(args.output)
    )
    print(
        f"Wrote {edge_rows} learned-score edge-ranking rows to {args.output} "
        f"and {summary_rows} summary rows to {summary_path}"
    )
    return 0


def _load_subject_blocks(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
) -> _LoadedSubjectBlocks:
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    sessions = tuple(_load_subject_sessions(subject_dir, config))
    _validate_reference_roi_indices(reference, sessions)
    blocks = collect_reference_pairwise_example_blocks(
        sessions,
        reference,
        session_edges=session_edge_pairs(len(sessions), max_gap=config.max_gap),
        options=_reference_training_options(config, feature_names),
    )
    return _LoadedSubjectBlocks(
        subject_dir=subject_dir,
        sessions=sessions,
        reference=reference,
        blocks=blocks,
    )


def _reference_training_options(
    config: Track2pBenchmarkConfig, feature_names: Sequence[str]
) -> ReferenceTrainingOptions:
    return ReferenceTrainingOptions(
        curated_only=config.curated_only,
        transform_type=config.transform_type,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        feature_names=tuple(feature_names),
        pairwise_cost_kwargs=_pairwise_cost_kwargs_for_learned_features(
            config.cost,
            config.pairwise_cost_kwargs,
            feature_names,
        ),
    )


def _pairwise_cost_kwargs_for_learned_features(
    cost: AssociationCost,
    overrides: Mapping[str, Any] | None,
    feature_names: Sequence[str],
) -> dict[str, Any] | None:
    if cost == "calibrated":
        kwargs = dict(overrides or {})
    else:
        kwargs = _pairwise_cost_kwargs_for_config(cost, overrides)
    return pairwise_cost_kwargs_for_calibration_features(kwargs, feature_names)


def _training_blocks(
    subjects: Sequence[_LoadedSubjectBlocks],
) -> tuple[ReferencePairwiseExamples, ...]:
    blocks: list[ReferencePairwiseExamples] = []
    for subject in subjects:
        blocks.extend(subject.blocks)
    if not blocks:
        raise ValueError("At least one training block is required")
    return tuple(blocks)


def _fit_learned_score_model(
    score_model: LearnedScoreModel,
    training_blocks: Sequence[ReferencePairwiseExamples],
    *,
    logistic_model_kwargs: Mapping[str, Any] | None,
    monotone_options: MonotoneRankerOptions | None,
) -> Any:
    if score_model == "logistic":
        features, labels = collect_candidate_limited_training_examples(
            training_blocks,
            options=CandidateHardNegativeOptions(),
        )
        return fit_logistic_association_model(
            features,
            labels,
            feature_names=training_blocks[0].feature_names,
            model_kwargs=logistic_model_kwargs,
        )
    if score_model == "monotone":
        return fit_monotone_ranking_association_model_from_blocks(
            training_blocks,
            options=monotone_options,
        )
    raise ValueError(f"Unsupported learned score model: {score_model!r}")


def _learned_score_matrices(
    model: Any,
    block: ReferencePairwiseExamples,
    *,
    score_model: LearnedScoreModel,
) -> dict[str, np.ndarray]:
    features = np.asarray(block.features, dtype=float)
    probabilities = np.asarray(model.predict_match_probability(features), dtype=float)
    costs = _cost_matrix_from_model(model, features, probabilities)
    if score_model == "logistic":
        return {
            "calibrated_cost": costs,
            "calibrated_probability": probabilities,
        }
    matrices = {
        "monotone_cost": costs,
        "monotone_probability": probabilities,
    }
    raw_score = getattr(model, "raw_cost_score", None)
    if callable(raw_score):
        matrices["monotone_raw_score"] = np.asarray(raw_score(features), dtype=float)
    return matrices


def _cost_matrix_from_model(
    model: Any, features: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    pairwise_cost_matrix = getattr(model, "pairwise_cost_matrix", None)
    if callable(pairwise_cost_matrix):
        return np.asarray(pairwise_cost_matrix(features), dtype=float)
    return -np.log(np.clip(probabilities, _EPSILON, 1.0))


def _score_directions_for_model(
    score_model: LearnedScoreModel,
) -> dict[str, ScoreDirection]:
    if score_model == "logistic":
        return {"calibrated_cost": "cost", "calibrated_probability": "similarity"}
    return {
        "monotone_cost": "cost",
        "monotone_probability": "similarity",
        "monotone_raw_score": "cost",
    }


def _config_from_args(args: argparse.Namespace) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = _json_object_or_none(args.pairwise_cost_kwargs_json)
    return Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="leave-one-subject-out",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=(
            args.allow_track2p_as_reference_for_smoke_test
        ),
        curated_only=args.curated_only,
        cost=args.cost,
        max_gap=args.max_gap,
        transform_type=args.transform_type,
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


def _monotone_options_from_args(
    args: argparse.Namespace,
) -> MonotoneRankerOptions | None:
    parsed = _json_object_or_none(args.monotone_ranker_kwargs_json)
    return None if parsed is None else MonotoneRankerOptions(**parsed)


def _json_object_or_none(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON option arguments must decode to an object")
    return parsed


def _unique_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(feature) for feature in feature_names))


def _validate_score_model(score_model: str) -> LearnedScoreModel:
    if score_model not in {"logistic", "monotone"}:
        raise ValueError("score_model must be 'logistic' or 'monotone'")
    return score_model  # type: ignore[return-value]


def _write_stdout(rows: Sequence[Mapping[str, float | int | str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
