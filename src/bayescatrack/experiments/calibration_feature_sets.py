"""Shared calibrated-association feature-set presets for Track2p experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from bayescatrack.association.calibrated_costs import (
    ACTIVITY_ASSOCIATION_FEATURES,
    DEFAULT_ASSOCIATION_FEATURES,
    DEFAULT_SHIFTED_OVERLAP_PAIRWISE_COST_KWARGS,
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    SHIFTED_OVERLAP_ASSOCIATION_FEATURES,
    SPLIT_ROI_STAT_FEATURES,
)

CalibrationFeatureSet = Literal[
    "default",
    "split-roi",
    "local-evidence",
    "default+split-roi",
    "default+local-evidence",
    "default+split-roi+local-evidence",
    "activity",
    "default+activity",
    "activity+local-evidence",
    "default+activity+local-evidence",
    "shifted-overlap",
    "default+shifted-overlap",
    "default+local-evidence+shifted-overlap",
    "rich",
]

CALIBRATION_FEATURE_SET_CHOICES: tuple[CalibrationFeatureSet, ...] = (
    "default",
    "split-roi",
    "local-evidence",
    "default+split-roi",
    "default+local-evidence",
    "default+split-roi+local-evidence",
    "activity",
    "default+activity",
    "activity+local-evidence",
    "default+activity+local-evidence",
    "shifted-overlap",
    "default+shifted-overlap",
    "default+local-evidence+shifted-overlap",
    "rich",
)

_LOCAL_EVIDENCE_COMPONENT_KWARGS = frozenset(
    {
        "local_evidence_components",
        "weighted_dice_weight",
        "overlap_fraction_weight",
        "containment_weight",
        "distance_transform_weight",
        "image_patch_weight",
        "neighbor_constellation_weight",
        "centroid_rank_weight",
    }
)


def calibration_feature_names(
    feature_set: CalibrationFeatureSet | str = "default",
) -> tuple[str, ...]:
    """Return the calibrated-association feature names for a named preset."""

    if feature_set == "default":
        return tuple(DEFAULT_ASSOCIATION_FEATURES)
    if feature_set == "split-roi":
        return tuple(SPLIT_ROI_STAT_FEATURES)
    if feature_set == "local-evidence":
        return tuple(LOCAL_EVIDENCE_ASSOCIATION_FEATURES)
    if feature_set == "default+split-roi":
        return _deduplicated_feature_names(
            DEFAULT_ASSOCIATION_FEATURES,
            SPLIT_ROI_STAT_FEATURES,
        )
    if feature_set == "default+local-evidence":
        return _deduplicated_feature_names(
            DEFAULT_ASSOCIATION_FEATURES,
            LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
        )
    if feature_set == "default+split-roi+local-evidence":
        return _deduplicated_feature_names(
            DEFAULT_ASSOCIATION_FEATURES,
            SPLIT_ROI_STAT_FEATURES,
            LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
        )
    if feature_set == "activity":
        return tuple(ACTIVITY_ASSOCIATION_FEATURES)
    if feature_set == "default+activity":
        return _deduplicated_feature_names(
            DEFAULT_ASSOCIATION_FEATURES,
            ACTIVITY_ASSOCIATION_FEATURES,
        )
    if feature_set == "activity+local-evidence":
        return _deduplicated_feature_names(
            ACTIVITY_ASSOCIATION_FEATURES,
            LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
        )
    if feature_set == "default+activity+local-evidence":
        return _deduplicated_feature_names(
            DEFAULT_ASSOCIATION_FEATURES,
            ACTIVITY_ASSOCIATION_FEATURES,
            LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
        )
    if feature_set == "shifted-overlap":
        return tuple(SHIFTED_OVERLAP_ASSOCIATION_FEATURES)
    if feature_set == "default+shifted-overlap":
        return _deduplicated_feature_names(
            DEFAULT_ASSOCIATION_FEATURES,
            SHIFTED_OVERLAP_ASSOCIATION_FEATURES,
        )
    if feature_set == "default+local-evidence+shifted-overlap":
        return _deduplicated_feature_names(
            DEFAULT_ASSOCIATION_FEATURES,
            LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
            SHIFTED_OVERLAP_ASSOCIATION_FEATURES,
        )
    if feature_set == "rich":
        return _deduplicated_feature_names(
            DEFAULT_ASSOCIATION_FEATURES,
            SPLIT_ROI_STAT_FEATURES,
            ACTIVITY_ASSOCIATION_FEATURES,
            LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
            SHIFTED_OVERLAP_ASSOCIATION_FEATURES,
        )
    raise ValueError(
        "calibration feature set must be one of: "
        + ", ".join(CALIBRATION_FEATURE_SET_CHOICES)
    )


def pairwise_cost_kwargs_for_calibration_features(
    pairwise_cost_kwargs: Mapping[str, Any] | None,
    feature_names: Sequence[str],
) -> dict[str, Any] | None:
    """Return pairwise-cost kwargs needed to materialize requested features."""

    kwargs = dict(pairwise_cost_kwargs or {})
    if uses_local_evidence_features(feature_names):
        kwargs.setdefault("local_evidence_components", True)
    if uses_shifted_overlap_features(feature_names):
        for key, value in DEFAULT_SHIFTED_OVERLAP_PAIRWISE_COST_KWARGS.items():
            kwargs.setdefault(key, value)
    return kwargs or None


def pairwise_kwargs_request_local_evidence(
    pairwise_cost_kwargs: Mapping[str, Any] | None,
) -> bool:
    """Return whether pairwise kwargs request local-evidence components."""

    if not pairwise_cost_kwargs:
        return False
    for key in _LOCAL_EVIDENCE_COMPONENT_KWARGS:
        value = pairwise_cost_kwargs.get(key)
        if isinstance(value, bool):
            if value:
                return True
        elif value is not None and float(value) > 0.0:
            return True
    return False


def uses_local_evidence_features(feature_names: Sequence[str]) -> bool:
    local_evidence_features = set(LOCAL_EVIDENCE_ASSOCIATION_FEATURES)
    return any(
        feature_name in local_evidence_features for feature_name in feature_names
    )


def uses_shifted_overlap_features(feature_names: Sequence[str]) -> bool:
    shifted_overlap_features = set(SHIFTED_OVERLAP_ASSOCIATION_FEATURES)
    return any(
        feature_name in shifted_overlap_features for feature_name in feature_names
    )


def _deduplicated_feature_names(*feature_groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(feature for group in feature_groups for feature in group)
    )
