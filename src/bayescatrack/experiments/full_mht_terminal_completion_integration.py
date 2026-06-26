"""Opt-in complete-history terminal objective for FullMHT.

The base FullMHT runner scores scan assignments locally and can optionally rerank
terminal hypotheses with label-free history penalties.  This module adds one more
terminal penalty: incomplete seed-anchored histories.  It is deliberately an
opt-in integration layer so frozen benchmark rows are unchanged unless a runner
or manifest attaches ``terminal_incomplete_history_weight`` to ``FullMHTConfig``.

The risk is label-free.  It counts missing observations in rows that contain at
least one observation, then multiplies that count by the configured weight.  A
complete-track-aware row can therefore prefer a slightly lower local scan score
when the alternative preserves more complete identity histories.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def install_full_mht_terminal_completion_objective() -> None:
    """Install the terminal incomplete-history objective into FullMHT."""

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    if getattr(full_mht, "_bayescatrack_terminal_completion_objective", False):
        return

    original_weight = full_mht._terminal_identity_history_weight
    original_risk = full_mht._terminal_identity_history_risk

    def _terminal_identity_history_weight_with_completion(config: Any) -> float:
        return max(
            float(original_weight(config)),
            _terminal_incomplete_history_weight(config),
        )

    def _terminal_identity_history_risk_with_completion(
        hypothesis: Any, *, config: Any
    ) -> float:
        risk = float(original_risk(hypothesis, config=config))
        weight = _terminal_incomplete_history_weight(config)
        if weight <= 0.0:
            return risk
        return float(risk + weight * terminal_incomplete_history_count(hypothesis))

    full_mht._terminal_identity_history_weight = (  # type: ignore[method-assign]
        _terminal_identity_history_weight_with_completion
    )
    full_mht._terminal_identity_history_risk = (  # type: ignore[method-assign]
        _terminal_identity_history_risk_with_completion
    )
    full_mht._bayescatrack_terminal_completion_original_weight = original_weight
    full_mht._bayescatrack_terminal_completion_original_risk = original_risk
    full_mht._bayescatrack_terminal_completion_objective = True


def terminal_incomplete_history_count(hypothesis: Any) -> int:
    """Count missing observations in non-empty terminal histories."""

    tracks = np.asarray(getattr(hypothesis, "tracks", hypothesis), dtype=int)
    if tracks.ndim != 2 or tracks.size == 0:
        return 0
    observed_rows = np.any(tracks >= 0, axis=1)
    if not np.any(observed_rows):
        return 0
    return int(np.sum((tracks[observed_rows] < 0).astype(int)))


def _terminal_incomplete_history_weight(config: Any) -> float:
    try:
        return max(0.0, float(getattr(config, "terminal_incomplete_history_weight", 0.0)))
    except (TypeError, ValueError):
        return 0.0
