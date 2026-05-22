from __future__ import annotations

from bayescatrack.experiments import (
    _calibration_feature_registry_integration as integration,
)
from bayescatrack.experiments import (
    track2p_benchmark,
    track2p_loso_calibration,
)
from bayescatrack.experiments.calibration_feature_sets import (
    CALIBRATION_FEATURE_SET_CHOICES,
)


def test_integration_patches_loso_feature_registry():
    integration.install_calibration_feature_registry_integration()

    assert (
        track2p_loso_calibration.CALIBRATION_FEATURE_SET_CHOICES
        == CALIBRATION_FEATURE_SET_CHOICES
    )
    assert "rich" in track2p_loso_calibration.CALIBRATION_FEATURE_SET_CHOICES
    assert "split-roi" in track2p_loso_calibration.CALIBRATION_FEATURE_SET_CHOICES
    assert "shifted_iou_cost" in track2p_loso_calibration.calibration_feature_names(
        "rich"
    )


def test_integration_patches_track2p_benchmark_parser_choices():
    integration.install_calibration_feature_registry_integration()

    args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--method",
            "global-assignment",
            "--split",
            "leave-one-subject-out",
            "--cost",
            "calibrated",
            "--calibration-feature-set",
            "rich",
            "--no-progress",
        ]
    )

    assert args.calibration_feature_set == "rich"
