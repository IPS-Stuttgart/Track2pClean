from __future__ import annotations

from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    DEFAULT_SHIFTED_OVERLAP_PAIRWISE_COST_KWARGS,
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    SHIFTED_OVERLAP_ASSOCIATION_FEATURES,
    SPLIT_ROI_STAT_FEATURES,
)
from bayescatrack.experiments.calibration_feature_sets import (
    CALIBRATION_FEATURE_SET_CHOICES,
    calibration_feature_names,
    pairwise_cost_kwargs_for_calibration_features,
    uses_local_evidence_features,
    uses_shifted_overlap_features,
)


def test_documented_feature_presets_are_available():
    assert "rich" in CALIBRATION_FEATURE_SET_CHOICES
    assert "split-roi" in CALIBRATION_FEATURE_SET_CHOICES


def test_feature_presets_are_deduplicated_and_stable():
    assert calibration_feature_names("default") == DEFAULT_ASSOCIATION_FEATURES
    assert calibration_feature_names("split-roi") == SPLIT_ROI_STAT_FEATURES
    assert len(calibration_feature_names("rich")) == len(
        set(calibration_feature_names("rich"))
    )


def test_single_calibration_feature_name_string_is_not_split_into_characters():
    local_feature = LOCAL_EVIDENCE_ASSOCIATION_FEATURES[0]
    shifted_feature = SHIFTED_OVERLAP_ASSOCIATION_FEATURES[0]

    assert uses_local_evidence_features(local_feature)
    assert uses_shifted_overlap_features(shifted_feature)
    assert pairwise_cost_kwargs_for_calibration_features(None, local_feature) == {
        "local_evidence_components": True,
    }
    assert (
        pairwise_cost_kwargs_for_calibration_features(None, shifted_feature)
        == DEFAULT_SHIFTED_OVERLAP_PAIRWISE_COST_KWARGS
    )
