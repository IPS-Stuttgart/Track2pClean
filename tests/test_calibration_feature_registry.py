from __future__ import annotations

from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    SPLIT_ROI_STAT_FEATURES,
)
from bayescatrack.experiments.calibration_feature_sets import (
    CALIBRATION_FEATURE_SET_CHOICES,
    calibration_feature_names,
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
