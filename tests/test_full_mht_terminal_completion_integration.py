from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht
from bayescatrack.experiments.full_mht_terminal_completion_integration import (
    install_full_mht_terminal_completion_objective,
    terminal_incomplete_history_count,
)
from bayescatrack.experiments.track2p_policy_full_mht_terminal_completion_benchmark import (
    _split_completion_args,
)


def test_terminal_incomplete_history_count_ignores_empty_rows() -> None:
    hypothesis = full_mht._MHTHypothesis(
        np.asarray(
            [
                [1, 2, 3, 4],
                [10, -1, 30, 40],
                [100, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=int,
        ),
        0.0,
        tuple(),
    )

    assert terminal_incomplete_history_count(hypothesis) == 4


def test_terminal_completion_weight_enters_identity_history_risk() -> None:
    install_full_mht_terminal_completion_objective()
    config = full_mht.FullMHTConfig()
    object.__setattr__(config, "terminal_incomplete_history_weight", 2.5)
    hypothesis = full_mht._MHTHypothesis(
        np.asarray([[1, 2, -1], [4, 5, 6]], dtype=int),
        0.0,
        tuple(),
    )

    assert full_mht._terminal_identity_history_weight(config) >= 2.5
    assert full_mht._terminal_identity_history_risk(hypothesis, config=config) == pytest.approx(2.5)


def test_terminal_completion_can_rerank_final_hypothesis() -> None:
    install_full_mht_terminal_completion_objective()
    config = full_mht.FullMHTConfig()
    object.__setattr__(config, "terminal_incomplete_history_weight", 2.0)
    locally_best_but_incomplete = full_mht._MHTHypothesis(
        np.asarray([[1, 2, -1]], dtype=int),
        10.0,
        tuple(),
    )
    lower_local_score_but_complete = full_mht._MHTHypothesis(
        np.asarray([[1, 2, 3]], dtype=int),
        9.0,
        tuple(),
    )

    selected, summary = full_mht._select_final_hypothesis(
        [locally_best_but_incomplete, lower_local_score_but_complete],
        sessions=tuple(),
        feature_cache=None,  # type: ignore[arg-type]
        config=config,
        track2p_prior_edges=frozenset(),
    )

    assert selected is lower_local_score_but_complete
    assert summary["terminal_selected_rank"] == 2
    assert summary["terminal_identity_history_risk"] == 0.0
    assert summary["terminal_adjusted_score"] == pytest.approx(9.0)


def test_terminal_completion_runner_splits_base_args() -> None:
    base_args, attrs = _split_completion_args(
        [
            "--data",
            "data-root",
            "--output",
            "out.csv",
            "--terminal-incomplete-history-weight",
            "1.25",
        ]
    )

    assert base_args == ["--data", "data-root", "--output", "out.csv"]
    assert attrs == {"terminal_incomplete_history_weight": 1.25}
