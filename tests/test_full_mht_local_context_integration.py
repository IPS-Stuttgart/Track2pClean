from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bayescatrack.experiments._full_mht_manifest_integration import (
    _uses_calibrated_association,
)
from bayescatrack.experiments.full_mht_local_context_integration import (
    install_full_mht_local_context_likelihood_gate,
)

full_mht = pytest.importorskip(
    "bayescatrack.experiments.track2p_policy_full_mht_benchmark"
)


def test_manifest_integration_detects_calibrated_local_context_rows():
    assert _uses_calibrated_association(
        {"association_score_mode": "calibrated-likelihood"}
    )
    assert not _uses_calibrated_association({"association_score_mode": "heuristic"})
    assert not _uses_calibrated_association({})


def test_local_context_likelihood_gate_zeros_feature_when_disabled(monkeypatch):
    observed: dict[str, np.ndarray] = {}

    def fake_likelihood(*, local_deformation, config, **_kwargs):
        observed["local_deformation"] = np.asarray(local_deformation, dtype=float)
        return observed["local_deformation"]

    monkeypatch.setattr(
        full_mht,
        "_bayescatrack_full_mht_local_context_gate",
        False,
        raising=False,
    )
    monkeypatch.setattr(full_mht, "_association_log_likelihood_matrix", fake_likelihood)

    install_full_mht_local_context_likelihood_gate()
    output = full_mht._association_log_likelihood_matrix(
        local_deformation=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        config=SimpleNamespace(local_deformation_weight=0.0),
    )

    assert np.array_equal(output, np.zeros((2, 2)))
    assert np.array_equal(observed["local_deformation"], np.zeros((2, 2)))


def test_local_context_likelihood_gate_preserves_feature_when_enabled(monkeypatch):
    observed: dict[str, np.ndarray] = {}

    def fake_likelihood(*, local_deformation, config, **_kwargs):
        observed["local_deformation"] = np.asarray(local_deformation, dtype=float)
        return observed["local_deformation"]

    local = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(
        full_mht,
        "_bayescatrack_full_mht_local_context_gate",
        False,
        raising=False,
    )
    monkeypatch.setattr(full_mht, "_association_log_likelihood_matrix", fake_likelihood)

    install_full_mht_local_context_likelihood_gate()
    output = full_mht._association_log_likelihood_matrix(
        local_deformation=local,
        config=SimpleNamespace(local_deformation_weight=0.5),
    )

    assert np.array_equal(output, local)
    assert np.array_equal(observed["local_deformation"], local)
