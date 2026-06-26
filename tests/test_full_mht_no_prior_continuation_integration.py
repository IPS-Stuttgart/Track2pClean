from __future__ import annotations

import numpy as np
import pytest


def _full_mht_module():
    pytest.importorskip("pyrecest")
    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht
    from bayescatrack.experiments.full_mht_no_prior_continuation_integration import (
        install_full_mht_no_prior_continuation_scoring,
    )

    install_full_mht_no_prior_continuation_scoring()
    return full_mht


def _matrices(full_mht):
    return full_mht._FullMHTPairMatrices(
        source_session=0,
        target_session=1,
        source_indices=np.asarray([5, 6], dtype=int),
        target_indices=np.asarray([9, 10], dtype=int),
        registered_iou=np.asarray([[0.91, 0.25], [0.30, 0.88]], dtype=float),
        shifted_iou=np.asarray([[0.80, 0.20], [0.25, 0.77]], dtype=float),
        centroid_distance=np.asarray([[1.0, 12.0], [11.0, 1.2]], dtype=float),
        area_ratio=np.asarray([[0.96, 0.50], [0.55, 0.94]], dtype=float),
        threshold=0.50,
        growth_residual=np.asarray([[0.20, 4.0], [3.8, 0.30]], dtype=float),
        growth_mahalanobis=np.asarray([[0.30, 5.0], [4.5, 0.40]], dtype=float),
        local_deformation=np.asarray([[0.02, 0.55], [0.45, 0.03]], dtype=float),
        growth_anchor_count=2,
        growth_model_type="affine",
    )


def _config(full_mht, *, weight: float):
    config = full_mht.FullMHTConfig(
        track2p_non_prior_penalty=0.0,
        track2p_no_prior_successor_penalty=0.0,
    )
    object.__setattr__(config, "no_prior_continuation_likelihood_weight", weight)
    object.__setattr__(config, "no_prior_continuation_min_examples_per_class", 1)
    return config


def test_no_prior_continuation_scoring_rewards_anchor_like_continuations(monkeypatch):
    full_mht = _full_mht_module()
    matrices = _matrices(full_mht)
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 0.95)
    prior_edges = frozenset({(0, 1, 99, 100)})

    base = _config(full_mht, weight=0.0)
    weighted = _config(full_mht, weight=1.0)
    base_anchor = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=base,
        track2p_prior_edges=prior_edges,
    )
    base_weak = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=1,
        config=base,
        track2p_prior_edges=prior_edges,
    )
    weighted_anchor = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=weighted,
        track2p_prior_edges=prior_edges,
    )
    weighted_weak = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=1,
        config=weighted,
        track2p_prior_edges=prior_edges,
    )

    assert weighted_anchor - base_anchor > 0.0
    assert weighted_weak - base_weak < 0.0


def test_no_prior_continuation_scoring_does_not_affect_prior_or_switch_edges(monkeypatch):
    full_mht = _full_mht_module()
    matrices = _matrices(full_mht)
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 0.95)
    config = _config(full_mht, weight=1.0)

    prior_edge = frozenset({(0, 1, 5, 9)})
    base_prior = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=_config(full_mht, weight=0.0),
        track2p_prior_edges=prior_edge,
    )
    weighted_prior = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=prior_edge,
    )

    switch_prior = frozenset({(0, 1, 5, 10)})
    base_switch = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=_config(full_mht, weight=0.0),
        track2p_prior_edges=switch_prior,
    )
    weighted_switch = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=switch_prior,
    )

    assert weighted_prior == pytest.approx(base_prior)
    assert weighted_switch == pytest.approx(base_switch)


def test_no_prior_continuation_summary_reports_weighted_score(monkeypatch):
    full_mht = _full_mht_module()
    matrices = _matrices(full_mht)
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 0.95)
    config = _config(full_mht, weight=1.0)

    summary = full_mht._selected_edge_summary(
        (object(), object()),
        matrices,
        active_source=full_mht._ActiveTrackSource(0, 0, 5, 0),
        target_session=1,
        target_roi=9,
        config=config,
        track2p_prior_edges=frozenset({(0, 1, 99, 100)}),
    )

    assert "no_prior_cont=" in summary["summary"]
    assert "no_prior_cont_weighted=" in summary["summary"]
    assert summary["no_prior_continuation_weighted_score"] > 0.0
