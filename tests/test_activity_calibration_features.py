from __future__ import annotations

from bayescatrack.association.activity_similarity import ACTIVITY_TIEBREAKER_FEATURES
from bayescatrack.association.calibrated_costs import (
    ACTIVITY_ASSOCIATION_FEATURES,
    DEFAULT_ASSOCIATION_FEATURES,
)
from bayescatrack.experiments import (
    track2p_benchmark,
    track2p_configurable_loso_calibration,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    calibration_feature_names,
)


def test_activity_calibration_feature_set_contains_rich_activity_features():
    names = calibration_feature_names("activity")

    assert names == ACTIVITY_ASSOCIATION_FEATURES
    assert names == tuple(ACTIVITY_TIEBREAKER_FEATURES)
    assert "fluorescence_similarity_cost" in names
    assert "spike_similarity_cost" in names
    assert "trace_std_absdiff" in names
    assert "trace_skew_absdiff" in names
    assert "event_rate_absdiff" in names
    assert "neuropil_ratio_absdiff" in names
    assert "activity_tiebreaker_available" in names


def test_default_activity_calibration_feature_set_keeps_default_prefix():
    names = calibration_feature_names("default+activity")

    assert names[: len(DEFAULT_ASSOCIATION_FEATURES)] == DEFAULT_ASSOCIATION_FEATURES
    assert len(names) == len(set(names))
    assert "activity_similarity_cost" in names
    assert "fluorescence_similarity_cost" in names
    assert "spike_similarity_cost" in names
    assert "event_rate_absdiff" in names


def test_activity_and_local_evidence_feature_sets_can_be_combined():
    names = calibration_feature_names("default+activity+local-evidence")

    assert "centroid_distance" in names
    assert "activity_tiebreaker_cost" in names
    assert len(names) == len(set(names))


# pylint: disable=protected-access
def test_track2p_benchmark_cli_accepts_activity_calibration_feature_set():
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
            "default+activity",
            "--no-progress",
        ]
    )

    config = track2p_benchmark._config_from_args(args)

    assert config.calibration_feature_set == "default+activity"


# pylint: disable=protected-access
def test_configurable_loso_cli_accepts_activity_calibration_feature_set():
    args = track2p_configurable_loso_calibration.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--calibration-feature-set",
            "activity",
            "--no-progress",
        ]
    )

    config = track2p_configurable_loso_calibration._config_from_args(args)

    assert config.calibration_feature_set == "activity"
    assert (
        calibration_feature_names(args.calibration_feature_set)
        == ACTIVITY_ASSOCIATION_FEATURES
    )
