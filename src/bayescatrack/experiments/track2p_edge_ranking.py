"""Export manual-GT edge-ranking diagnostics for Track2p-style datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bayescatrack.association.calibrated_costs import (
    ReferenceTrainingOptions,
    collect_reference_pairwise_example_blocks,
    fit_logistic_association_model,
)
from bayescatrack.association.monotone_ranking_costs import (
    MonotoneRankerOptions,
    fit_monotone_ranked_association_model,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    registered_iou_cost_kwargs,
    registered_shifted_iou_cost_kwargs,
    roi_aware_shifted_cost_kwargs,
    roi_aware_cost_kwargs,
    session_edge_pairs,
)
from bayescatrack.evaluation.edge_ranking import (
    DEFAULT_HIT_KS,
    ScoreDirection,
    missing_reference_edge_rows,
    rank_labeled_edges,
    score_matrices_from_feature_tensor,
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
from bayescatrack.soft_overlap_costs import registered_soft_iou_cost_kwargs

DEFAULT_EDGE_RANKING_FEATURES = (
    "pairwise_cost_matrix",
    "iou",
    "iou_cost",
    "centroid_distance",
    "mahalanobis_centroid_distance",
    "mask_cosine_similarity",
    "mask_cosine_cost",
    "area_ratio_cost",
    "covariance_shape_cost",
    "covariance_logdet_cost",
    "roi_feature_cost",
    "cell_probability_cost",
    "activity_similarity",
    "activity_similarity_cost",
)

DEFAULT_SIMILARITY_FEATURES = (
    "iou",
    "mask_cosine_similarity",
    "activity_similarity",
)

LearnedScoreModel = Literal["none", "logistic", "monotone"]

EDGE_FIELDNAMES = [
    "subject",
    "session_a",
    "session_b",
    "session_a_name",
    "session_b_name",
    "session_gap",
    "learned_score_model",
    "training_subjects",
    "reference_roi_index",
    "measurement_roi_index",
    "score_name",
    "score_direction",
    "edge_present",
    "missing_reason",
    "true_score",
    "true_is_finite",
    "row_rank",
    "column_rank",
    "row_better_count",
    "column_better_count",
    "row_tie_count",
    "column_tie_count",
    "row_candidate_count",
    "column_candidate_count",
    "row_finite_candidate_count",
    "column_finite_candidate_count",
    "best_false_row_score",
    "best_false_column_score",
    "best_false_row_roi_index",
    "best_false_column_roi_index",
    "row_margin",
    "column_margin",
]

SUMMARY_FIELDNAMES = [
    "subject",
    "session_a",
    "session_b",
    "session_gap",
    "score_name",
    "gt_edges",
    "present_edges",
    "missing_edges",
    "finite_true_edges",
    "row_hit_at_1",
    "row_hit_at_3",
    "row_hit_at_5",
    "row_hit_at_10",
    "column_hit_at_1",
    "column_hit_at_3",
    "column_hit_at_5",
    "column_hit_at_10",
    "row_hit_at_1_present",
    "row_hit_at_3_present",
    "row_hit_at_5_present",
    "row_hit_at_10_present",
    "column_hit_at_1_present",
    "column_hit_at_3_present",
    "column_hit_at_5_present",
    "column_hit_at_10_present",
    "mutual_top1_rate",
    "mutual_top1_rate_present",
    "median_row_rank",
    "median_column_rank",
    "median_row_margin",
    "median_column_margin",
    "mean_row_margin",
    "mean_column_margin",
    "row_positive_margin_rate",
    "column_positive_margin_rate",
]


def run_track2p_edge_ranking(
    config: Track2pBenchmarkConfig,
    output_path: Path,
    *,
    summary_output_path: Path | None = None,
    feature_names: Sequence[str] = DEFAULT_EDGE_RANKING_FEATURES,
    similarity_features: Sequence[str] = DEFAULT_SIMILARITY_FEATURES,
    hit_ks: Sequence[int] = DEFAULT_HIT_KS,
    learned_score_model: LearnedScoreModel = "none",
    calibrated_model_kwargs: Mapping[str, Any] | None = None,
    monotone_ranker_options: MonotoneRankerOptions | None = None,
) -> tuple[int, int]:
    """Export per-edge and per-session-pair edge-ranking diagnostics.

    The detailed CSV contains one row per manual-GT edge per score. The summary
    CSV groups those rows by subject, session pair, gap, and score name.
    """

    learned_score_model = _effective_learned_score_model(
        learned_score_model, config.cost
    )
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    feature_names = tuple(dict.fromkeys(str(feature) for feature in feature_names))
    if not feature_names:
        raise ValueError("At least one feature/score name is required")

    if learned_score_model != "none" and len(subject_dirs) < 2:
        raise ValueError("Learned-score edge ranking requires at least two subjects")

    score_directions: dict[str, ScoreDirection] = {
        str(feature): "similarity" for feature in similarity_features
    }
    output_rows: list[dict[str, float | int | str]] = []
    progress = ProgressReporter(
        len(subject_dirs), enabled=config.progress, label="edge-ranking"
    )

    for subject_dir in subject_dirs:
        progress.step(f"ranking {subject_dir.name}")
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=config
        )
        sessions = _load_subject_sessions(subject_dir, config)
        _validate_reference_roi_indices(reference, sessions)
        learned_model = None
        training_subject_names: tuple[str, ...] = ()
        if learned_score_model != "none":
            training_subject_names = tuple(
                candidate.name for candidate in subject_dirs if candidate != subject_dir
            )
            training_blocks = _collect_learned_score_training_blocks(
                subject_dirs,
                subject_dir,
                config=config,
                feature_names=feature_names,
            )
            learned_model = _fit_learned_score_model(
                training_blocks,
                learned_score_model=learned_score_model,
                feature_names=feature_names,
                calibrated_model_kwargs=calibrated_model_kwargs,
                monotone_ranker_options=monotone_ranker_options,
            )
        options = _reference_training_options(config, feature_names)
        edges = session_edge_pairs(len(sessions), max_gap=config.max_gap)
        blocks = collect_reference_pairwise_example_blocks(
            sessions, reference, session_edges=edges, options=options
        )
        for block in blocks:
            metadata = {
                "subject": subject_dir.name,
                "session_a": int(block.session_a),
                "session_b": int(block.session_b),
                "session_a_name": sessions[block.session_a].session_name,
                "session_b_name": sessions[block.session_b].session_name,
                "session_gap": int(block.gap),
            }
            if learned_score_model != "none":
                metadata["learned_score_model"] = learned_score_model
                metadata["training_subjects"] = ",".join(training_subject_names)
            score_matrices = score_matrices_from_feature_tensor(
                block.features, block.feature_names
            )
            if learned_model is not None:
                score_matrices.update(
                    _learned_score_matrices(
                        learned_model,
                        block.features,
                        learned_score_model=learned_score_model,
                    )
                )
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
                    reference.pairwise_matches(
                        block.session_a,
                        block.session_b,
                        curated_only=config.curated_only,
                    ),
                    reference_roi_indices=block.reference_roi_indices,
                    measurement_roi_indices=block.measurement_roi_indices,
                    score_names=tuple(score_matrices.keys()),
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


def _collect_learned_score_training_blocks(
    subject_dirs: Sequence[Path],
    held_out_subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    feature_names: Sequence[str],
) -> tuple[Any, ...]:
    """Collect manual-GT pairwise blocks from all non-held-out subjects."""

    blocks: list[Any] = []
    options = _reference_training_options(config, feature_names)
    for training_subject_dir in subject_dirs:
        if training_subject_dir == held_out_subject_dir:
            continue
        reference = _load_reference_for_subject(
            training_subject_dir, data_root=config.data, config=config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=training_subject_dir, config=config
        )
        sessions = _load_subject_sessions(training_subject_dir, config)
        _validate_reference_roi_indices(reference, sessions)
        blocks.extend(
            collect_reference_pairwise_example_blocks(
                sessions,
                reference,
                session_edges=session_edge_pairs(
                    len(sessions), max_gap=config.max_gap
                ),
                options=options,
            )
        )
    if not blocks:
        raise ValueError("No learned-score training blocks were collected")
    return tuple(blocks)


def _fit_learned_score_model(
    training_blocks: Sequence[Any],
    *,
    learned_score_model: LearnedScoreModel,
    feature_names: Sequence[str],
    calibrated_model_kwargs: Mapping[str, Any] | None,
    monotone_ranker_options: MonotoneRankerOptions | None,
) -> Any:
    """Fit the LOSO model whose learned scores will be ranked on held-out data."""

    if learned_score_model == "logistic":
        features, labels = collect_candidate_limited_training_examples(
            training_blocks, options=CandidateHardNegativeOptions()
        )
        model_kwargs = {"class_weight": None}
        model_kwargs.update(dict(calibrated_model_kwargs or {}))
        return fit_logistic_association_model(
            features,
            labels,
            feature_names=tuple(feature_names),
            model_kwargs=model_kwargs,
        )
    if learned_score_model == "monotone":
        return fit_monotone_ranked_association_model(
            training_blocks,
            feature_names=tuple(feature_names),
            options=monotone_ranker_options or MonotoneRankerOptions(),
        )
    raise ValueError(f"Unsupported learned-score model: {learned_score_model!r}")


def _learned_score_matrices(
    calibrated_model: Any,
    features: Any,
    *,
    learned_score_model: LearnedScoreModel,
) -> dict[str, Any]:
    """Return learned cost/probability score planes for one feature tensor."""

    prefix = "calibrated" if learned_score_model == "logistic" else "monotone"
    probabilities = _predict_probability_matrix(calibrated_model, features)
    matrices: dict[str, Any] = {
        f"{prefix}_match_probability": probabilities,
        f"{prefix}_cost": -np.log(np.clip(probabilities, 1.0e-12, 1.0)),
    }
    raw_score = _optional_raw_score_matrix(calibrated_model, features)
    if raw_score is not None:
        matrices[f"{prefix}_raw_score"] = raw_score
    return matrices


def _predict_probability_matrix(calibrated_model: Any, features: Any) -> np.ndarray:
    if hasattr(calibrated_model, "predict_match_probability"):
        return _matrix_from_feature_predictor(
            calibrated_model.predict_match_probability, features
        )
    model = getattr(calibrated_model, "model", calibrated_model)
    if hasattr(model, "predict_match_probability"):
        return _matrix_from_feature_predictor(model.predict_match_probability, features)
    if hasattr(model, "predict_proba"):
        return _matrix_from_feature_predictor(
            lambda values: np.asarray(model.predict_proba(values), dtype=float)[
                ..., -1
            ],
            features,
        )
    if hasattr(model, "pairwise_cost_matrix"):
        costs = _matrix_from_feature_predictor(model.pairwise_cost_matrix, features)
        return np.exp(-np.clip(costs, 0.0, 1.0e12))
    raise TypeError("Learned model does not expose probability or cost prediction")


def _optional_raw_score_matrix(
    calibrated_model: Any, features: Any
) -> np.ndarray | None:
    model = getattr(calibrated_model, "model", calibrated_model)
    for method_name in ("predict_score", "raw_cost_score"):
        if hasattr(model, method_name):
            return _matrix_from_feature_predictor(getattr(model, method_name), features)
    return None


def _matrix_from_feature_predictor(predictor: Any, features: Any) -> np.ndarray:
    feature_array = np.asarray(features, dtype=float)
    expected_shape = feature_array.shape[:-1]
    try:
        values = np.asarray(predictor(feature_array), dtype=float)
    except (TypeError, ValueError):
        values = np.empty((), dtype=float)
    if values.shape == expected_shape:
        return values
    flat_values = np.asarray(
        predictor(feature_array.reshape(-1, feature_array.shape[-1])), dtype=float
    )
    if flat_values.ndim >= 1 and flat_values.shape[-1] == 1:
        flat_values = flat_values[..., 0]
    return flat_values.reshape(expected_shape)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the edge-ranking diagnostic."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark edge-ranking",
        description="Export manual-GT pairwise edge-ranking diagnostics before global assignment.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Detailed per-GT-edge CSV output path",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Summary CSV path; defaults to <output-stem>_summary.csv",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional ground_truth.csv file, ground-truth root, subject directory, or track2p folder",
    )
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
        "--cost",
        default="registered-iou",
        choices=(
            "registered-iou",
            "registered-soft-iou",
            "registered-shifted-iou",
            "roi-aware",
            "roi-aware-shifted",
            "calibrated",
            "monotone",
        ),
        help=(
            "Raw pairwise cost to rank; calibrated/monotone enable "
            "LOSO learned scores"
        ),
    )
    parser.add_argument(
        "--learned-score-model",
        default="none",
        choices=("none", "logistic", "monotone"),
    )
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
    parser.add_argument(
        "--pairwise-cost-kwargs-json",
        default=None,
        help="JSON object merged into pairwise cost kwargs before ranking pairwise_cost_matrix",
    )
    parser.add_argument(
        "--calibrated-model-kwargs-json",
        default=None,
        help="JSON object passed to the logistic model for learned-score ranking",
    )
    parser.add_argument(
        "--monotone-ranker-kwargs-json",
        default=None,
        help="JSON object used to construct MonotoneRankerOptions",
    )
    parser.add_argument(
        "--feature",
        dest="features",
        action="append",
        default=None,
        help="Feature/component to rank; repeat to override the default feature set",
    )
    parser.add_argument(
        "--similarity-feature",
        dest="similarity_features",
        action="append",
        default=None,
        help="Declare a ranked feature where larger values are better; repeat as needed",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress to stderr",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    feature_names = (
        tuple(args.features)
        if args.features is not None
        else DEFAULT_EDGE_RANKING_FEATURES
    )
    learned_score_model = _effective_learned_score_model(
        args.learned_score_model, args.cost
    )
    similarity_features = (
        tuple(args.similarity_features)
        if args.similarity_features is not None
        else DEFAULT_SIMILARITY_FEATURES
    )
    calibrated_model_kwargs = _json_object_or_none(
        args.calibrated_model_kwargs_json, "--calibrated-model-kwargs-json"
    )
    monotone_ranker_options = _monotone_ranker_options_from_args(args)
    edge_rows, summary_rows = run_track2p_edge_ranking(
        config,
        args.output,
        summary_output_path=args.summary_output,
        feature_names=feature_names,
        similarity_features=similarity_features,
        learned_score_model=learned_score_model,
        calibrated_model_kwargs=calibrated_model_kwargs,
        monotone_ranker_options=monotone_ranker_options,
    )
    summary_path = (
        args.summary_output
        if args.summary_output is not None
        else _default_summary_output_path(args.output)
    )
    print(
        f"Wrote {edge_rows} edge-ranking rows to {args.output} and {summary_rows} summary rows to {summary_path}"
    )
    return 0


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
        pairwise_cost_kwargs=_pairwise_cost_kwargs_for_config(
            _edge_ranking_base_cost(config.cost), config.pairwise_cost_kwargs
        ),
    )


def _pairwise_cost_kwargs_for_config(
    cost: AssociationCost,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cost = _edge_ranking_base_cost(cost)
    kwargs: dict[str, Any]
    if cost == "registered-iou":
        kwargs = registered_iou_cost_kwargs()
    elif cost == "registered-soft-iou":
        kwargs = registered_soft_iou_cost_kwargs()
    elif cost == "registered-shifted-iou":
        kwargs = registered_shifted_iou_cost_kwargs()
    elif cost == "roi-aware":
        kwargs = roi_aware_cost_kwargs()
    elif cost == "roi-aware-shifted":
        kwargs = roi_aware_shifted_cost_kwargs()
    else:
        raise ValueError(f"Unsupported edge-ranking cost: {cost!r}")
    if overrides is not None:
        kwargs.update(dict(overrides))
    return kwargs


def _effective_learned_score_model(requested: str, cost: str) -> LearnedScoreModel:
    if requested == "logistic":
        return "logistic"
    if requested == "monotone":
        return "monotone"
    if requested != "none":
        raise ValueError("learned_score_model must be one of: none, logistic, monotone")
    if cost == "calibrated":
        return "logistic"
    if cost == "monotone":
        return "monotone"
    return "none"


def _edge_ranking_base_cost(cost: str) -> str:
    if cost in {"calibrated", "monotone"}:
        return "registered-iou"
    return cost


def _json_object_or_none(value: str | None, option_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return parsed


def _monotone_ranker_options_from_args(
    args: argparse.Namespace,
) -> MonotoneRankerOptions | None:
    parsed = _json_object_or_none(
        args.monotone_ranker_kwargs_json, "--monotone-ranker-kwargs-json"
    )
    return None if parsed is None else MonotoneRankerOptions(**parsed)


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
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        cost=_edge_ranking_base_cost(args.cost),
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


def _default_summary_output_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    suffix = output_path.suffix or ".csv"
    return output_path.with_name(f"{output_path.stem}_summary{suffix}")


def _write_csv(
    rows: Sequence[Mapping[str, float | int | str]],
    output_path: Path,
    *,
    preferred_fieldnames: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows, preferred_fieldnames)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(
    rows: Sequence[Mapping[str, float | int | str]], preferred: Sequence[str]
) -> list[str]:
    row_keys = {key for row in rows for key in row}
    return [key for key in preferred if key in row_keys] + sorted(
        row_keys - set(preferred)
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
