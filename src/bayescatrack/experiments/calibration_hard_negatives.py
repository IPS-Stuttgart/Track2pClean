"""Candidate-limited hard-negative sampling for Track2p calibration folds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from bayescatrack.association.calibrated_costs import (
    CalibratedAssociationModel,
    ReferencePairwiseExamples,
)

_HARD_NEGATIVE_DEFAULT_FEATURES = (
    "mahalanobis_centroid_distance",
    "centroid_distance",
    "one_minus_iou",
    "one_minus_mask_cosine",
    "area_ratio_cost",
    "covariance_shape_cost",
    "covariance_logdet_cost",
    "roi_feature_cost",
    "cell_probability_cost",
    "activity_similarity_cost",
)


@dataclass(frozen=True)
class CandidateHardNegativeOptions:
    """Candidate-limited hard-negative selection for calibrated training."""

    negative_to_positive_ratio: float = 4.0
    candidate_top_k_per_anchor: int | None = 20
    include_column_candidates: bool = True
    hardness_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ratio = float(self.negative_to_positive_ratio)
        if not np.isfinite(ratio) or ratio < 0.0:
            raise ValueError(
                "negative_to_positive_ratio must be finite and non-negative"
            )
        object.__setattr__(self, "negative_to_positive_ratio", ratio)
        if self.candidate_top_k_per_anchor is not None:
            top_k = int(self.candidate_top_k_per_anchor)
            if top_k <= 0:
                raise ValueError("candidate_top_k_per_anchor must be positive or None")
            object.__setattr__(self, "candidate_top_k_per_anchor", top_k)
        object.__setattr__(
            self,
            "hardness_feature_names",
            tuple(
                ()
                if self.hardness_feature_names is None
                else self.hardness_feature_names
            ),
        )


def collect_candidate_limited_training_examples(
    example_blocks: Sequence[ReferencePairwiseExamples],
    *,
    options: CandidateHardNegativeOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all positives and a bounded set of hard negatives from pairwise blocks."""

    options = options or CandidateHardNegativeOptions()
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for block in example_blocks:
        features, labels = _validated_pairwise_block_arrays(block)
        selected_mask = _candidate_hard_negative_selection_mask(block, options)
        selected_labels = labels[selected_mask].reshape(-1)
        if selected_labels.size == 0:
            continue
        feature_blocks.append(features[selected_mask].reshape(-1, features.shape[-1]))
        label_blocks.append(selected_labels)
    if not feature_blocks:
        raise ValueError("No calibration training examples were selected")
    return np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks, axis=0)


def balanced_binary_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Return inverse-frequency weights for binary calibration labels."""

    labels = np.asarray(labels, dtype=int).reshape(-1)
    weights = np.ones(labels.shape, dtype=float)
    n_examples = int(labels.size)
    if n_examples == 0:
        return weights
    positive = labels != 0
    negative = ~positive
    for mask in (positive, negative):
        n_class = int(np.sum(mask))
        if n_class:
            weights[mask] = n_examples / (2.0 * n_class)
    return weights


def _validated_pairwise_block_arrays(
    block: ReferencePairwiseExamples,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(block.features, dtype=float)
    labels = np.asarray(block.labels, dtype=int)
    if features.ndim != 3:
        raise ValueError("Pairwise training features must be three-dimensional")
    if labels.shape != features.shape[:2]:
        raise ValueError("Pairwise labels must match the first two feature dimensions")
    if features.shape[-1] != len(block.feature_names):
        raise ValueError("Pairwise feature-name count does not match feature tensor")
    return features, labels


def _candidate_hard_negative_selection_mask(
    block: ReferencePairwiseExamples,
    options: CandidateHardNegativeOptions,
) -> np.ndarray:
    _features, labels = _validated_pairwise_block_arrays(block)
    positive_mask = labels != 0
    candidate_mask = _top_k_candidate_negative_mask(block, options)
    selected_negative_mask = np.zeros(labels.shape, dtype=bool)
    n_positive = int(np.sum(positive_mask))
    if n_positive > 0 and options.negative_to_positive_ratio > 0.0:
        max_negatives = int(np.ceil(options.negative_to_positive_ratio * n_positive))
        flat_candidates = np.flatnonzero(candidate_mask.reshape(-1))
        if flat_candidates.size:
            hardness = _pairwise_hardness_score(
                block, hardness_feature_names=options.hardness_feature_names
            )
            order = np.lexsort((flat_candidates, hardness.reshape(-1)[flat_candidates]))
            selected_flat = flat_candidates[order[:max_negatives]]
            selected_negative_mask.reshape(-1)[selected_flat] = True
    return positive_mask | selected_negative_mask


def _top_k_candidate_negative_mask(
    block: ReferencePairwiseExamples,
    options: CandidateHardNegativeOptions,
) -> np.ndarray:
    _features, labels = _validated_pairwise_block_arrays(block)
    candidate_mask = labels == 0
    if options.candidate_top_k_per_anchor is None or not np.any(candidate_mask):
        return candidate_mask

    hardness = _pairwise_hardness_score(
        block, hardness_feature_names=options.hardness_feature_names
    )
    top_mask = np.zeros_like(candidate_mask)
    top_k = int(options.candidate_top_k_per_anchor)
    for row_index in range(candidate_mask.shape[0]):
        candidate_cols = np.flatnonzero(candidate_mask[row_index])
        if candidate_cols.size:
            ordered_cols = _ordered_candidate_indices(
                hardness[row_index, candidate_cols], candidate_cols
            )
            top_mask[row_index, ordered_cols[:top_k]] = True
    if options.include_column_candidates:
        for col_index in range(candidate_mask.shape[1]):
            candidate_rows = np.flatnonzero(candidate_mask[:, col_index])
            if candidate_rows.size:
                ordered_rows = _ordered_candidate_indices(
                    hardness[candidate_rows, col_index], candidate_rows
                )
                top_mask[ordered_rows[:top_k], col_index] = True
    return top_mask


def _ordered_candidate_indices(scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return indices[np.lexsort((indices, np.asarray(scores, dtype=float)))]


def _pairwise_hardness_score(
    block: ReferencePairwiseExamples,
    *,
    hardness_feature_names: Sequence[str] = (),
) -> np.ndarray:
    feature_names = tuple(block.feature_names)
    if hardness_feature_names:
        selected_names = tuple(hardness_feature_names)
        missing = [name for name in selected_names if name not in feature_names]
        if missing:
            raise ValueError(
                "Hard-negative feature names are missing from the feature tensor: "
                + ", ".join(missing)
            )
    else:
        selected_names = tuple(
            name for name in _HARD_NEGATIVE_DEFAULT_FEATURES if name in feature_names
        )
        if not selected_names:
            selected_names = tuple(
                name
                for name in feature_names
                if name != "session_gap" and not name.endswith("_available")
            )
    if not selected_names:
        return np.zeros(np.asarray(block.labels).shape, dtype=float)
    planes = [
        _robust_normalized_cost_plane(_required_feature_plane(block, feature_name))
        for feature_name in selected_names
    ]
    return np.mean(np.stack(planes, axis=0), axis=0)


def _required_feature_plane(
    block: ReferencePairwiseExamples, feature_name: str
) -> np.ndarray:
    feature_names = tuple(block.feature_names)
    if feature_name not in feature_names:
        raise ValueError(f"Feature {feature_name!r} is required for hard negatives")
    features, _labels = _validated_pairwise_block_arrays(block)
    return features[..., feature_names.index(feature_name)]


def _robust_normalized_cost_plane(values: np.ndarray) -> np.ndarray:
    plane = np.nan_to_num(
        np.asarray(values, dtype=float), nan=0.0, posinf=1.0e6, neginf=-1.0e6
    )
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros_like(plane, dtype=float)
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    return (plane - center) / scale


@dataclass(frozen=True)
class MonotoneRankerTrainingOptions:
    """Projected ranking-loss options for hard-negative calibrated costs."""

    hard_negative_options: CandidateHardNegativeOptions = field(
        default_factory=CandidateHardNegativeOptions
    )
    margin: float = 1.0
    l2_regularization: float = 1.0e-3
    learning_rate: float = 0.1
    max_iter: int = 400
    probability_temperature: float | None = None
    max_constraints_per_positive: int | None = 6

    def __post_init__(self) -> None:
        margin = float(self.margin)
        if not np.isfinite(margin) or margin <= 0.0:
            raise ValueError("margin must be finite and strictly positive")
        object.__setattr__(self, "margin", margin)

        l2_regularization = float(self.l2_regularization)
        if not np.isfinite(l2_regularization) or l2_regularization < 0.0:
            raise ValueError("l2_regularization must be finite and non-negative")
        object.__setattr__(self, "l2_regularization", l2_regularization)

        learning_rate = float(self.learning_rate)
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and strictly positive")
        object.__setattr__(self, "learning_rate", learning_rate)

        max_iter = int(self.max_iter)
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        object.__setattr__(self, "max_iter", max_iter)

        if self.probability_temperature is not None:
            probability_temperature = float(self.probability_temperature)
            if (
                not np.isfinite(probability_temperature)
                or probability_temperature <= 0.0
            ):
                raise ValueError(
                    "probability_temperature must be finite and strictly positive"
                )
            object.__setattr__(self, "probability_temperature", probability_temperature)

        if self.max_constraints_per_positive is not None:
            max_constraints = int(self.max_constraints_per_positive)
            if max_constraints <= 0:
                raise ValueError(
                    "max_constraints_per_positive must be positive or None"
                )
            object.__setattr__(self, "max_constraints_per_positive", max_constraints)


@dataclass(frozen=True)
class MonotoneHardNegativeRanker:
    """Non-negative linear ranking model exposed as calibrated pairwise costs.

    The model learns a cost function in which lower feature values are better and
    constrains all learned weights to be non-negative. Training uses only
    fold-local hard-negative comparisons, so the decision boundary is driven by
    whether a reference-positive ROI pair outranks same-row/same-column
    distractors rather than by the overwhelming easy-negative class prior.
    """

    feature_names: tuple[str, ...]
    feature_center: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    cost_offset: float
    probability_temperature: float

    def _raw_cost(self, features: object) -> np.ndarray:
        transformed = _ranker_feature_matrix(
            np.asarray(features, dtype=float),
            center=np.asarray(self.feature_center, dtype=float),
            scale=np.asarray(self.feature_scale, dtype=float),
        )
        return np.nan_to_num(
            np.tensordot(
                transformed,
                np.asarray(self.weights, dtype=float),
                axes=([-1], [0]),
            ),
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
        )

    def pairwise_cost_matrix(self, features: object) -> np.ndarray:
        """Return non-negative assignment costs for feature vectors or tensors."""

        decision = (self._raw_cost(features) - float(self.cost_offset)) / float(
            self.probability_temperature
        )
        return np.nan_to_num(
            np.logaddexp(0.0, decision), nan=1.0e6, posinf=1.0e6, neginf=0.0
        )

    def predict_match_probability(self, features: object) -> np.ndarray:
        """Return match probabilities implied by the monotone ranking cost."""

        decision = (float(self.cost_offset) - self._raw_cost(features)) / float(
            self.probability_temperature
        )
        return _sigmoid(decision)


def fit_monotone_hard_negative_ranker(
    example_blocks: Sequence[ReferencePairwiseExamples],
    *,
    feature_names: Sequence[str] | None = None,
    options: MonotoneRankerTrainingOptions | None = None,
) -> CalibratedAssociationModel:
    """Fit a monotone hard-negative ranker and wrap it as a calibrated model."""

    blocks = tuple(example_blocks)
    if not blocks:
        raise ValueError("At least one pairwise training block is required")
    options = options or MonotoneRankerTrainingOptions()
    feature_names = _ranker_feature_names(blocks, feature_names)
    candidate_features, candidate_labels = collect_candidate_limited_training_examples(
        blocks, options=options.hard_negative_options
    )
    candidate_labels = np.asarray(candidate_labels, dtype=int).reshape(-1)
    if not np.any(candidate_labels != 0) or not np.any(candidate_labels == 0):
        raise ValueError(
            "Monotone ranker training requires positive and negative examples"
        )

    feature_center, feature_scale = _robust_feature_center_and_scale(candidate_features)
    constraints = _ranking_constraint_matrix(
        blocks,
        feature_names=feature_names,
        center=feature_center,
        scale=feature_scale,
        options=options,
    )
    if constraints.size:
        weights = _fit_projected_hinge_weights(
            constraints,
            feature_names=feature_names,
            margin=options.margin,
            l2_regularization=options.l2_regularization,
            learning_rate=options.learning_rate,
            max_iter=options.max_iter,
        )
    else:
        weights = _fallback_monotone_weights(
            candidate_features,
            candidate_labels,
            feature_names=feature_names,
            center=feature_center,
            scale=feature_scale,
        )
    if not np.any(weights > 0.0):
        weights = _fallback_monotone_weights(
            candidate_features,
            candidate_labels,
            feature_names=feature_names,
            center=feature_center,
            scale=feature_scale,
        )

    transformed_training = _ranker_feature_matrix(
        candidate_features, center=feature_center, scale=feature_scale
    )
    raw_scores = transformed_training @ weights
    cost_offset = _ranker_cost_offset(raw_scores, candidate_labels)
    temperature = (
        float(options.probability_temperature)
        if options.probability_temperature is not None
        else _ranker_probability_temperature(raw_scores)
    )
    model = MonotoneHardNegativeRanker(
        feature_names=feature_names,
        feature_center=feature_center,
        feature_scale=feature_scale,
        weights=weights,
        cost_offset=cost_offset,
        probability_temperature=temperature,
    )
    return CalibratedAssociationModel(model=model, feature_names=feature_names)


def _ranker_feature_names(
    blocks: Sequence[ReferencePairwiseExamples],
    feature_names: Sequence[str] | None,
) -> tuple[str, ...]:
    names = tuple(blocks[0].feature_names if feature_names is None else feature_names)
    if not names:
        raise ValueError("At least one feature is required")
    for block in blocks:
        if tuple(block.feature_names) != names:
            raise ValueError(
                "All ranking training blocks must use the same feature names"
            )
    return names


# pylint: disable=too-many-locals
def _ranking_constraint_matrix(
    blocks: Sequence[ReferencePairwiseExamples],
    *,
    feature_names: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
    options: MonotoneRankerTrainingOptions,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for block in blocks:
        features, labels = _validated_pairwise_block_arrays(block)
        candidate_mask = _top_k_candidate_negative_mask(
            block, options.hard_negative_options
        )
        hardness = _pairwise_hardness_score(
            block,
            hardness_feature_names=options.hard_negative_options.hardness_feature_names,
        )
        transformed = _ranker_feature_matrix(features, center=center, scale=scale)
        positive_positions = np.argwhere(labels != 0)
        for row_index, col_index in positive_positions:
            negative_positions = _ranking_negative_positions(
                candidate_mask,
                int(row_index),
                int(col_index),
                include_column_candidates=(
                    options.hard_negative_options.include_column_candidates
                ),
            )
            if not negative_positions:
                continue
            negative_positions = _limit_negative_positions(
                negative_positions,
                hardness,
                max_constraints=options.max_constraints_per_positive,
            )
            positive_features = transformed[int(row_index), int(col_index)]
            for negative_row, negative_col in negative_positions:
                rows.append(transformed[negative_row, negative_col] - positive_features)
    if not rows:
        return np.zeros((0, len(tuple(feature_names))), dtype=float)
    return np.nan_to_num(
        np.stack(rows, axis=0), nan=0.0, posinf=1.0e6, neginf=-1.0e6
    )


def _ranking_negative_positions(
    candidate_mask: np.ndarray,
    row_index: int,
    col_index: int,
    *,
    include_column_candidates: bool,
) -> list[tuple[int, int]]:
    positions: dict[tuple[int, int], None] = {}
    for negative_col in np.flatnonzero(candidate_mask[row_index]):
        positions[(row_index, int(negative_col))] = None
    if include_column_candidates:
        for negative_row in np.flatnonzero(candidate_mask[:, col_index]):
            positions[(int(negative_row), col_index)] = None
    positions.pop((row_index, col_index), None)
    return list(positions)


def _limit_negative_positions(
    positions: Sequence[tuple[int, int]],
    hardness: np.ndarray,
    *,
    max_constraints: int | None,
) -> list[tuple[int, int]]:
    if max_constraints is None or len(positions) <= max_constraints:
        return list(positions)
    ordered = sorted(
        positions,
        key=lambda pos: (float(hardness[pos[0], pos[1]]), int(pos[0]), int(pos[1])),
    )
    return ordered[:max_constraints]


def _robust_feature_center_and_scale(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=float).reshape(
        -1, np.asarray(features).shape[-1]
    )
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    center = np.median(matrix, axis=0)
    absolute_deviation = np.abs(matrix - center[None, :])
    scale = 1.4826 * np.median(absolute_deviation, axis=0)
    std = np.std(matrix, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    return center.astype(float), scale.astype(float)


def _ranker_feature_matrix(
    features: np.ndarray, *, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.shape[-1] != center.shape[0]:
        raise ValueError("Feature matrix last dimension does not match ranker schema")
    finite_values = np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return (finite_values - center) / scale


def _fit_projected_hinge_weights(
    constraints: np.ndarray,
    *,
    feature_names: Sequence[str],
    margin: float,
    l2_regularization: float,
    learning_rate: float,
    max_iter: int,
) -> np.ndarray:
    n_features = int(constraints.shape[1])
    mask = _ranker_weight_mask(feature_names)
    weights = mask / max(float(np.sum(mask)), 1.0)
    for iteration in range(int(max_iter)):
        margins = constraints @ weights
        violation = np.maximum(0.0, float(margin) - margins)
        if np.any(violation > 0.0):
            active = violation > 0.0
            gradient = -(
                violation[active, None] * constraints[active]
            ).mean(axis=0)
        else:
            gradient = np.zeros((n_features,), dtype=float)
        gradient += float(l2_regularization) * weights
        step = float(learning_rate) / np.sqrt(1.0 + iteration / 50.0)
        weights = np.maximum(0.0, weights - step * gradient) * mask
    total = float(np.sum(weights))
    if np.isfinite(total) and total > 0.0:
        weights = weights / total
    return np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


def _fallback_monotone_weights(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    transformed = _ranker_feature_matrix(features, center=center, scale=scale)
    positive = transformed[np.asarray(labels, dtype=int) != 0]
    negative = transformed[np.asarray(labels, dtype=int) == 0]
    if positive.size and negative.size:
        weights = np.maximum(
            np.median(negative, axis=0) - np.median(positive, axis=0), 0.0
        )
    else:
        weights = np.ones((transformed.shape[-1],), dtype=float)
    weights = np.asarray(weights, dtype=float) * _ranker_weight_mask(feature_names)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        weights = _ranker_weight_mask(feature_names)
        total = float(np.sum(weights))
    return weights / max(total, 1.0)


def _ranker_weight_mask(feature_names: Sequence[str]) -> np.ndarray:
    mask = np.ones((len(tuple(feature_names)),), dtype=float)
    for index, feature_name in enumerate(feature_names):
        if feature_name == "session_gap" or feature_name.endswith("_available"):
            mask[index] = 0.0
    if not np.any(mask > 0.0):
        mask[:] = 1.0
    return mask


def _ranker_cost_offset(raw_scores: np.ndarray, labels: np.ndarray) -> float:
    raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    positive_scores = raw_scores[labels != 0]
    negative_scores = raw_scores[labels == 0]
    if positive_scores.size and negative_scores.size:
        return float(
            0.5 * (np.median(positive_scores) + np.median(negative_scores))
        )
    return float(np.median(raw_scores)) if raw_scores.size else 0.0


def _ranker_probability_temperature(raw_scores: np.ndarray) -> float:
    scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    if scores.size <= 1:
        return 1.0
    q75, q25 = np.percentile(scores, [75.0, 25.0])
    temperature = float((q75 - q25) / 1.349)
    if not np.isfinite(temperature) or temperature <= 1.0e-6:
        temperature = float(np.std(scores))
    if not np.isfinite(temperature) or temperature <= 1.0e-6:
        temperature = 1.0
    return temperature


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))
