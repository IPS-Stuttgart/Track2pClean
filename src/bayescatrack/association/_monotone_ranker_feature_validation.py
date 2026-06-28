"""Validation for monotone-ranker feature-name schemas.

The monotone rankers map requested feature names back to tensor columns with
``tuple.index``. Duplicate feature names are therefore ambiguous: a duplicate in
``ReferencePairwiseExamples.feature_names`` silently resolves to the first column,
and a duplicate requested monotone feature silently double-counts one column.
Fail fast before training so calibration experiments cannot depend on ambiguous
feature schemas.

Feature tensors are validated before model prediction so scalar inputs fail with a
user-facing ``ValueError`` instead of an internal tuple-index error from
``shape[-1]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_MARKER = "_track2pclean_monotone_ranker_feature_validation"
_RANKING_COSTS_MARKER = "_track2pclean_monotone_ranking_cost_feature_validation"
_FEATURE_ARRAY_MARKER = "_track2pclean_monotone_ranker_feature_array_validation"
_FEATURE_ARRAY_ERROR = "features must be an array with a final feature dimension"


def install_monotone_ranker_feature_validation() -> None:
    """Install idempotent validation around monotone-ranker training and prediction."""

    from . import (
        monotone_ranker as _monotone_ranker,  # pylint: disable=import-outside-toplevel
    )
    from . import (
        monotone_ranking_costs as _monotone_ranking_costs,  # pylint: disable=import-outside-toplevel
    )

    original = _monotone_ranker.fit_monotone_ranking_association_model_from_blocks
    if not getattr(original, _MARKER, False):

        @wraps(original)
        def fit_monotone_ranking_association_model_from_blocks_with_feature_validation(
            example_blocks: Sequence[Any],
            *,
            options: Any | None = None,
        ) -> Any:
            blocks = tuple(example_blocks)
            for block in blocks:
                _validate_unique_feature_names(
                    getattr(block, "feature_names", ()),
                    field_name="feature_names",
                )
            if options is not None:
                _validate_unique_feature_names(
                    getattr(options, "monotone_feature_names", ()),
                    field_name="monotone_feature_names",
                )
            return original(blocks, options=options)

        setattr(
            fit_monotone_ranking_association_model_from_blocks_with_feature_validation,
            _MARKER,
            True,
        )
        setattr(
            fit_monotone_ranking_association_model_from_blocks_with_feature_validation,
            "_track2pclean_original",
            original,
        )
        _monotone_ranker.fit_monotone_ranking_association_model_from_blocks = (
            fit_monotone_ranking_association_model_from_blocks_with_feature_validation
        )

    _patch_monotone_ranking_cost_fit(_monotone_ranking_costs)
    _patch_normalized_features(_monotone_ranker.MonotoneRankingAssociationModel)
    _patch_normalized_features(_monotone_ranking_costs.MonotonePairwiseRanker)


def _patch_monotone_ranking_cost_fit(module: Any) -> None:
    original = module.fit_monotone_ranked_association_model
    if getattr(original, _RANKING_COSTS_MARKER, False):
        return

    @wraps(original)
    def fit_monotone_ranked_association_model_with_feature_validation(
        example_blocks: Sequence[Any],
        *,
        feature_names: Sequence[str] | str | None = None,
        options: Any | None = None,
    ) -> Any:
        blocks = tuple(example_blocks)
        for block in blocks:
            _validate_unique_feature_names(
                getattr(block, "feature_names", ()),
                field_name="block feature_names",
            )
        if feature_names is not None:
            _validate_unique_feature_names(feature_names, field_name="feature_names")
        if options is not None:
            _validate_unique_feature_names(
                getattr(options, "hardness_feature_names", ()),
                field_name="hardness_feature_names",
            )
        return original(blocks, feature_names=feature_names, options=options)

    setattr(
        fit_monotone_ranked_association_model_with_feature_validation,
        _RANKING_COSTS_MARKER,
        True,
    )
    setattr(
        fit_monotone_ranked_association_model_with_feature_validation,
        "_track2pclean_original",
        original,
    )
    module.fit_monotone_ranked_association_model = (
        fit_monotone_ranked_association_model_with_feature_validation
    )


def _patch_normalized_features(model_cls: type[Any]) -> None:
    original = model_cls._normalized_features
    if getattr(original, _FEATURE_ARRAY_MARKER, False):
        return

    @wraps(original)
    def _normalized_features_with_array_validation(
        self: Any, features: Any
    ) -> np.ndarray:
        _validate_feature_tensor_has_dimension(features)
        return original(self, features)

    setattr(_normalized_features_with_array_validation, _FEATURE_ARRAY_MARKER, True)
    setattr(
        _normalized_features_with_array_validation, "_track2pclean_original", original
    )
    model_cls._normalized_features = _normalized_features_with_array_validation


def _validate_feature_tensor_has_dimension(features: Any) -> None:
    try:
        feature_array = np.asarray(features)
    except ValueError as exc:
        raise ValueError(_FEATURE_ARRAY_ERROR) from exc
    if feature_array.ndim == 0:
        raise ValueError(_FEATURE_ARRAY_ERROR)


def _feature_name_tuple(values: Any) -> tuple[Any, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    try:
        return tuple(values)
    except TypeError as exc:
        raise ValueError(
            "feature names must be provided as a sequence of feature names"
        ) from exc


def _validate_unique_feature_names(values: Any, *, field_name: str) -> None:
    names = _feature_name_tuple(values)

    seen: set[str] = set()
    duplicate_names: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{field_name} must contain non-empty string feature names"
            )
        if name in seen and name not in duplicate_names:
            duplicate_names.append(name)
        seen.add(name)

    if duplicate_names:
        duplicates = ", ".join(duplicate_names)
        raise ValueError(
            f"{field_name} must be unique; duplicate feature(s): {duplicates}"
        )


__all__ = ["install_monotone_ranker_feature_validation"]
