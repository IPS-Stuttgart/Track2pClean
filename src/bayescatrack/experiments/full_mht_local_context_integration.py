"""Local-context integration hooks for FullMHT calibrated likelihoods.

The core FullMHT runner includes local-neighborhood deformation as one feature in
the calibrated association likelihood.  This hook makes the existing
``local_deformation_weight`` knob a true ablation switch for manifest runs: when
the weight is zero or negative, the calibrated likelihood receives a zero local
context feature while all other association features remain unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def install_full_mht_local_context_likelihood_gate() -> None:
    """Install the calibrated-likelihood local-context ablation gate."""

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    if getattr(full_mht, "_bayescatrack_full_mht_local_context_gate", False):
        return

    original = full_mht._association_log_likelihood_matrix

    def _association_log_likelihood_matrix_with_local_context_gate(
        *,
        local_deformation: Any,
        config: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        local_feature = np.asarray(local_deformation, dtype=float)
        if float(getattr(config, "local_deformation_weight", 1.0)) <= 0.0:
            local_feature = np.zeros_like(local_feature, dtype=float)
        return original(
            local_deformation=local_feature,
            config=config,
            **kwargs,
        )

    full_mht._association_log_likelihood_matrix = (  # type: ignore[method-assign]
        _association_log_likelihood_matrix_with_local_context_gate
    )
    full_mht._bayescatrack_full_mht_local_context_gate = True
