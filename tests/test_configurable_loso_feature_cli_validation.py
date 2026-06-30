from __future__ import annotations

import pytest
from bayescatrack.experiments import track2p_configurable_loso_calibration as loso


def _args(extra: list[str]):
    return loso.build_arg_parser().parse_args(["--data", "dataset", *extra])


def test_configurable_loso_rejects_empty_comma_feature_tokens():
    args = _args(["--calibration-features", "centroid_distance,,one_minus_iou"])

    with pytest.raises(ValueError, match="--calibration-features"):
        loso._resolved_calibration_feature_names(args)


def test_configurable_loso_rejects_empty_repeated_feature():
    args = _args(["--calibration-feature", ""])

    with pytest.raises(ValueError, match="--calibration-feature"):
        loso._resolved_calibration_feature_names(args)


def test_configurable_loso_explicit_features_preserve_order_and_deduplicate():
    args = _args(
        [
            "--calibration-features",
            "centroid_distance,one_minus_iou",
            "--calibration-feature",
            "centroid_distance",
            "--calibration-feature",
            "session_gap",
        ]
    )

    assert loso._resolved_calibration_feature_names(args) == (
        "centroid_distance",
        "one_minus_iou",
        "session_gap",
    )
