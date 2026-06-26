"""Validation for monotone-ranker feature-name schemas.

The monotone ranker maps requested feature names back to tensor columns with
``tuple.index``. Duplicate feature names are therefore ambiguous: a duplicate in
``ReferencePairwiseExamples.feature_names`` silently resolves to the first column,
and a duplicate requested monotone feature silently double-counts one column.
Fail fast before training so calibration experiments cannot depend on ambiguous
feature schemas.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

_MARKER = "_track2pclean_monotone_ranker_feature_validation"


def install_monotone_ranker_feature_validation() -> None:
    """Install idempotent validation around monotone-ranker training."""

    from . import monotone_ranker as _monotone_ranker  # pylint: disable=import-outside-toplevel

    original = _monotone_ranker.fit_monotone_ranking_association_model_from_blocks
    if getattr(original, _MARKER, False):
        return

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


def _validate_unique_feature_names(values: Any, *, field_name: str) -> None:
    try:
        names = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of feature names") from exc

    seen: set[str] = set()
    duplicate_names: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field_name} must contain non-empty string feature names")
        if name in seen and name not in duplicate_names:
            duplicate_names.append(name)
        seen.add(name)

    if duplicate_names:
        duplicates = ", ".join(duplicate_names)
        raise ValueError(f"{field_name} must be unique; duplicate feature(s): {duplicates}")


__all__ = ["install_monotone_ranker_feature_validation"]
