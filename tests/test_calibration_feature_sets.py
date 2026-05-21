from __future__ import annotations

from bayescatrack.association.calibrated_costs import (
    ACTIVITY_ASSOCIATION_FEATURES,
    DEFAULT_ASSOCIATION_FEATURES,
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    SHIFTED_OVERLAP_ASSOCIATION_FEATURES,
    SPLIT_ROI_STAT_FEATURES,
)
from bayescatrack.experiments.calibration_feature_sets import (
    CALIBRATION_FEATURE_SET_CHOICES,
    calibration_feature_names,
    pairwise_cost_kwargs_for_calibration_features,
)


def test_feature_set_registry_contains_documented_rich_preset():
    assert "rich" in CALIBRATION_FEATURE_SET_CHOICES

    names = calibration_feature_names("rich")

    assert len(names) == len(set(names))
    for required in (
        "centroid_distance",
        "activity_tiebreaker_cost",
        "weighted_dice_cost",
        "shifted_iou_cost",
    ):
        assert required in names


def test_split_roi_feature_set_is_explicit_alias_for_suite2p_stat_features():
    assert calibration_feature_names("split-roi") == SPLIT_ROI_STAT_FEATURES
    assert calibration_feature_names("default") == DEFAULT_ASSOCIATION_FEATURES

    combined = calibration_feature_names("default+split-roi")
    assert combined[: len(DEFAULT_ASSOCIATION_FEATURES)] == DEFAULT_ASSOCIATION_FEATURES
    assert len(combined) == len(set(combined))


def test_registry_supports_existing_activity_and_shifted_overlap_presets():
    assert calibration_feature_names("activity") == ACTIVITY_ASSOCIATION_FEATURES

    rich = calibration_feature_names("rich")
    for group in (
        DEFAULT_ASSOCIATION_FEATURES,
        ACTIVITY_ASSOCIATION_FEATURES,
        LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
        SHIFTED_OVERLAP_ASSOCIATION_FEATURES,
    ):
        for feature in group:
            assert feature in rich


def test_pairwise_cost_kwargs_are_enabled_for_feature_materialization():
    names = calibration_feature_names("default+local-evidence+shifted-overlap")
    kwargs = pairwise_cost_kwargs_for_calibration_features(None, names)

    assert kwargs is not None
    assert kwargs["local_evidence_components"] is True
    assert kwargs["shifted_iou_radius"] == 2


def test_pairwise_cost_kwargs_preserve_user_overrides():
    names = calibration_feature_names("rich")
    kwargs = pairwise_cost_kwargs_for_calibration_features(
        {"shifted_iou_radius": 4, "local_evidence_components": False}, names
    )

    assert kwargs is not None
    assert kwargs["shifted_iou_radius"] == 4
    assert kwargs["local_evidence_components"] is False
