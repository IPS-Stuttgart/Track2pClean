from __future__ import annotations

import pytest
import numpy as np
from bayescatrack import cli
from bayescatrack.experiments.track2p_policy_stability_cleanup import (
    StabilityCleanupConfig,
    apply_stability_splits_to_tracks,
    edge_support_counts,
)


def test_edge_support_counts_counts_each_edge_once_per_prediction() -> None:
    predictions = (
        np.asarray([[10, 20, 30], [10, 20, 30]], dtype=int),
        np.asarray([[10, 20, -1]], dtype=int),
        np.asarray([[10, 21, 31]], dtype=int),
    )

    counts = edge_support_counts(predictions)

    assert counts[(0, 1, 10, 20)] == 2
    assert counts[(1, 2, 20, 30)] == 1
    assert counts[(0, 1, 10, 21)] == 1


def test_stability_cleanup_splits_low_support_bridge() -> None:
    base = np.asarray([[10, 20, 30, 40]], dtype=int)
    support = {
        (0, 1, 10, 20): 3,
        (1, 2, 20, 30): 1,
        (2, 3, 30, 40): 3,
    }

    cleaned, split_rows = apply_stability_splits_to_tracks(
        base,
        support,
        required_support_votes=2,
        min_side_observations=2,
    )

    np.testing.assert_array_equal(
        cleaned,
        np.asarray([[10, 20, -1, -1], [-1, -1, 30, 40]], dtype=int),
    )
    assert split_rows == (
        {
            "predicted_track_id": 0,
            "split_session_a": 1,
            "split_session_b": 2,
            "source_roi": 20,
            "target_roi": 30,
            "support_votes": 1,
            "required_support_votes": 2,
        },
    )


def test_stability_cleanup_keeps_stable_track() -> None:
    base = np.asarray([[10, 20, 30]], dtype=int)
    support = {
        (0, 1, 10, 20): 2,
        (1, 2, 20, 30): 2,
    }

    cleaned, split_rows = apply_stability_splits_to_tracks(
        base,
        support,
        required_support_votes=2,
        min_side_observations=2,
    )

    np.testing.assert_array_equal(cleaned, base)
    assert split_rows == ()


def test_stability_cleanup_rejects_splits_that_create_short_fragments() -> None:
    base = np.asarray([[10, 20, 30, 40, 50]], dtype=int)
    support = {
        (0, 1, 10, 20): 1,
        (1, 2, 20, 30): 1,
        (2, 3, 30, 40): 3,
        (3, 4, 40, 50): 3,
    }

    cleaned, split_rows = apply_stability_splits_to_tracks(
        base,
        support,
        required_support_votes=2,
        min_side_observations=2,
    )

    np.testing.assert_array_equal(
        cleaned,
        np.asarray([[10, 20, -1, -1, -1], [-1, -1, 30, 40, 50]], dtype=int),
    )
    assert len(split_rows) == 1
    assert split_rows[0]["split_session_a"] == 1


def test_stability_cleanup_config_includes_base_threshold_in_vote_ensemble() -> None:
    config = StabilityCleanupConfig(
        iou_distance_thresholds=(10.0, 14.0),
        base_iou_distance_threshold=12.0,
    )

    assert config.ensemble_iou_distance_thresholds == (12.0, 10.0, 14.0)
    assert config.required_support_votes == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_support_votes": True},
        {"min_support_votes": 1.5},
        {"min_support_votes": 0},
        {"min_side_observations": False},
        {"min_side_observations": 1.5},
        {"min_side_observations": 0},
    ],
)
def test_stability_cleanup_config_rejects_invalid_integer_options(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        StabilityCleanupConfig(**kwargs)


@pytest.mark.parametrize(
    ("required_support_votes", "min_side_observations"),
    [
        (True, 2),
        (1.5, 2),
        (0, 2),
        (2, False),
        (2, 1.5),
        (2, 0),
    ],
)
def test_apply_stability_splits_rejects_invalid_integer_options(
    required_support_votes: object, min_side_observations: object
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        apply_stability_splits_to_tracks(
            np.asarray([[10, 20]], dtype=int),
            {(0, 1, 10, 20): 1},
            required_support_votes=required_support_votes,  # type: ignore[arg-type]
            min_side_observations=min_side_observations,  # type: ignore[arg-type]
        )


def test_stability_cleanup_benchmark_command_is_registered() -> None:
    assert cli._BENCHMARK_COMMANDS["track2p-policy-stability-cleanup"].module == "bayescatrack.experiments.track2p_policy_stability_cleanup"
    assert cli._BENCHMARK_ALIASES["track2p-stability-cleanup"] == "track2p-policy-stability-cleanup"
