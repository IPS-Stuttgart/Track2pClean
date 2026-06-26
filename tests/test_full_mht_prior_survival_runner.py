from __future__ import annotations

from dataclasses import dataclass

from bayescatrack.experiments import (
    track2p_policy_full_mht_prior_survival_benchmark as runner,
)


def test_prior_survival_runner_splits_survival_args() -> None:
    base_argv, attrs = runner._split_survival_args(
        [
            "--data",
            "data-root",
            "--output",
            "out.csv",
            "--track2p-prior-survival-weight",
            "1.5",
            "--track2p-prior-survival-score-clip=4.0",
            "--track2p-prior-survival-min-examples-per-class",
            "3",
        ]
    )

    assert base_argv == ["--data", "data-root", "--output", "out.csv"]
    assert attrs["track2p_prior_survival_weight"] == 1.5
    assert attrs["track2p_prior_survival_score_clip"] == 4.0
    assert attrs["track2p_prior_survival_min_examples_per_class"] == 3


def test_prior_survival_runner_defaults_enable_canonical_row() -> None:
    base_argv, attrs = runner._split_survival_args(
        ["--data", "data-root", "--output", "out.csv"]
    )

    assert base_argv == ["--data", "data-root", "--output", "out.csv"]
    assert attrs["track2p_prior_survival_weight"] == 1.0
    assert attrs["track2p_prior_survival_min_examples_per_class"] == 2
    assert attrs["track2p_prior_survival_score_clip"] == 8.0


@dataclass(frozen=True)
class _FrozenConfig:
    base_weight: float = 12.0


def test_prior_survival_runner_attaches_attrs_to_frozen_config() -> None:
    config = _FrozenConfig()

    updated = runner._attach_survival_attrs(
        config,
        {
            "track2p_prior_survival_weight": 1.25,
            "track2p_prior_survival_min_examples_per_class": 2,
        },
    )

    assert updated is config
    assert updated.base_weight == 12.0
    assert getattr(updated, "track2p_prior_survival_weight") == 1.25
    assert getattr(updated, "track2p_prior_survival_min_examples_per_class") == 2
