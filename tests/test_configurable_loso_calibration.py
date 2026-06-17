from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.experiments import track2p_configurable_loso_calibration as loso
from bayescatrack.experiments import track2p_monotone_loso_calibration as monotone_loso
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_loso_calibration import calibration_feature_names

# pylint: disable=protected-access,too-few-public-methods


def test_configurable_loso_parser_builds_hard_negative_options_and_model_kwargs():
    args = loso.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--hard-negative-ratio",
            "7.5",
            "--hard-negative-top-k",
            "none",
            "--no-hard-negative-column-candidates",
            "--hard-negative-features",
            "centroid_distance,one_minus_iou",
            "--sample-weight-strategy",
            "balanced",
            "--calibration-model",
            "hist-gradient-boosting",
            "--calibration-model-kwargs-json",
            '{"max_iter":25}',
        ]
    )

    options = loso._hard_negative_options(args)

    assert args.sample_weight_strategy == "balanced"
    assert args.calibration_model == "hist-gradient-boosting"
    assert loso._json_object(args.calibration_model_kwargs_json, "model") == {
        "max_iter": 25
    }
    assert options.negative_to_positive_ratio == 7.5
    assert options.candidate_top_k_per_anchor is None
    assert not options.include_column_candidates
    assert options.hardness_feature_names == ("centroid_distance", "one_minus_iou")


def test_sklearn_probability_adapter_preserves_pairwise_tensor_shape():
    class DummyEstimator:
        def __init__(self):
            self.classes_ = None
            self.n_fit_examples = 0
            self.received_sample_weight = None

        def fit(self, features, _labels, **kwargs):
            self.classes_ = np.array([0, 1], dtype=int)
            self.n_fit_examples = int(features.shape[0])
            self.received_sample_weight = kwargs.get("sample_weight")
            return self

        def predict_proba(self, features):
            probabilities = np.linspace(0.1, 0.9, num=features.shape[0])
            return np.column_stack([1.0 - probabilities, probabilities])

    adapter = loso.SklearnPairwiseProbabilityAdapter(DummyEstimator())
    features = np.zeros((2, 3, 4), dtype=float)
    labels = np.array([[1, 0, 0], [0, 1, 0]], dtype=int)
    sample_weight = np.ones(labels.size, dtype=float)

    adapter.fit(features, labels, sample_weight=sample_weight)
    probabilities = adapter.predict_match_probability(features)

    assert adapter.estimator.n_fit_examples == 6
    assert adapter.estimator.received_sample_weight.shape == (6,)
    assert probabilities.shape == (2, 3)
    assert np.isclose(probabilities[0, 0], 0.1)
    assert np.isclose(probabilities[-1, -1], 0.9)


def test_resolved_feature_names_uses_config_default_when_unspecified():
    class DummyConfig:
        calibration_feature_set = "default"

    resolved = loso._resolved_feature_names(DummyConfig(), None)

    assert resolved == loso.calibration_feature_names("default")
    assert len(resolved) > 0


def test_configurable_loso_uses_config_calibration_defaults(monkeypatch, tmp_path):
    captured = {}

    def fake_validate_sample_weight_strategy(strategy):
        captured["sample_weight_strategy"] = strategy
        return strategy

    def fake_hard_negative_options_from_config(config):
        captured["hard_negative_config"] = config
        return loso.CandidateHardNegativeOptions(
            negative_to_positive_ratio=7.5,
            candidate_top_k_per_anchor=None,
            include_column_candidates=False,
        )

    def fake_load_subjects(config):
        captured["config"] = config
        raise ValueError("stop before loading subjects")

    monkeypatch.setattr(
        loso, "_validate_sample_weight_strategy", fake_validate_sample_weight_strategy
    )
    monkeypatch.setattr(
        loso,
        "_candidate_hard_negative_options_from_config",
        fake_hard_negative_options_from_config,
    )
    monkeypatch.setattr(loso, "_load_subjects", fake_load_subjects)
    config = Track2pBenchmarkConfig(
        data=tmp_path,
        method="global-assignment",
        split="leave-one-subject-out",
        cost="calibrated",
        calibration_sample_weight_strategy="balanced",
        calibration_hard_negative_ratio=7.5,
        calibration_candidate_top_k_per_anchor=None,
        calibration_include_column_candidates=False,
        progress=False,
    )

    with pytest.raises(ValueError, match="stop before loading subjects"):
        loso.run_track2p_configurable_loso_calibration(config)

    assert captured["sample_weight_strategy"] == "balanced"
    assert captured["hard_negative_config"] is config
    assert captured["config"].calibration_sample_weight_strategy == "balanced"


def test_monotone_loso_uses_config_calibration_feature_set(monkeypatch, tmp_path):
    captured = {}

    def fake_pairwise_cost_kwargs(existing_kwargs, feature_names):
        captured["feature_names"] = tuple(feature_names)
        return existing_kwargs

    monkeypatch.setattr(
        monotone_loso,
        "pairwise_cost_kwargs_for_calibration_features",
        fake_pairwise_cost_kwargs,
    )
    config = Track2pBenchmarkConfig(
        data=tmp_path,
        method="global-assignment",
        split="leave-one-subject-out",
        cost="calibrated",
        calibration_feature_set="activity",
        progress=False,
    )

    with pytest.raises(ValueError, match="LOSO calibration requires"):
        monotone_loso.run_track2p_monotone_loso_calibration(config)

    assert captured["feature_names"] == calibration_feature_names("activity")
