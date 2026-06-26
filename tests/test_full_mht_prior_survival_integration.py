from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyrecest")

from bayescatrack.experiments import (  # noqa: E402
    track2p_policy_full_mht_benchmark as full_mht,
)
from bayescatrack.experiments.full_mht_prior_survival_integration import (  # noqa: E402
    install_full_mht_prior_survival_scoring,
)


def _original_edge_score():
    return getattr(
        full_mht,
        "_bayescatrack_prior_survival_original_edge_score",
        full_mht._edge_score,
    )


def _prior_matrix() -> full_mht._FullMHTPairMatrices:
    diagonal = np.asarray([0.90, 0.88, 0.36, 0.42], dtype=float)
    shape = (4, 4)
    registered = np.full(shape, 0.01, dtype=float)
    shifted = np.full(shape, 0.01, dtype=float)
    growth_residual = np.full(shape, 4.0, dtype=float)
    growth_mahalanobis = np.full(shape, 4.0, dtype=float)
    area_ratio = np.full(shape, 0.50, dtype=float)
    local_deformation = np.full(shape, 0.50, dtype=float)
    np.fill_diagonal(registered, diagonal)
    np.fill_diagonal(
        shifted,
        np.asarray([0.80, 0.78, 0.76, 0.62], dtype=float),
    )
    np.fill_diagonal(
        growth_residual,
        np.asarray([0.20, 0.40, 2.90, 2.60]),
    )
    np.fill_diagonal(
        growth_mahalanobis,
        np.asarray([0.30, 0.60, 2.70, 3.20]),
    )
    np.fill_diagonal(area_ratio, np.asarray([0.98, 0.95, 0.86, 0.78]))
    np.fill_diagonal(local_deformation, np.asarray([0.02, 0.04, 0.25, 0.30]))
    return full_mht._FullMHTPairMatrices(
        source_session=0,
        target_session=1,
        source_indices=np.asarray([10, 20, 30, 40], dtype=int),
        target_indices=np.asarray([11, 21, 31, 41], dtype=int),
        registered_iou=registered,
        shifted_iou=shifted,
        centroid_distance=np.ones(shape, dtype=float),
        area_ratio=area_ratio,
        threshold=0.10,
        growth_residual=growth_residual,
        growth_mahalanobis=growth_mahalanobis,
        local_deformation=local_deformation,
        growth_anchor_count=2,
        growth_model_type="test",
    )


def _prior_edges() -> frozenset[tuple[int, int, int, int]]:
    return frozenset(
        {
            (0, 1, 10, 11),
            (0, 1, 20, 21),
            (0, 1, 30, 31),
            (0, 1, 40, 41),
        }
    )


def _patch_cell_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    cell_probability = {
        10: 0.93,
        11: 0.92,
        20: 0.90,
        21: 0.89,
        30: 0.62,
        31: 0.63,
        40: 0.66,
        41: 0.67,
    }
    monkeypatch.setattr(
        full_mht,
        "_cell_probability",
        lambda _sessions, _session, roi: cell_probability[int(roi)],
    )


def _score(edge_score, matrices, config, *, source_local: int) -> float:
    return edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=source_local,
        target_local=source_local,
        config=config,
        track2p_prior_edges=_prior_edges(),
    )


def test_prior_survival_scoring_rewards_anchors_and_penalizes_hazards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cell_probabilities(monkeypatch)
    matrices = _prior_matrix()
    install_full_mht_prior_survival_scoring()
    original = _original_edge_score()
    config = full_mht.FullMHTConfig(track2p_prior_weight=0.0)
    object.__setattr__(config, "track2p_prior_survival_weight", 1.0)

    anchor_score = _score(full_mht._edge_score, matrices, config, source_local=0)
    anchor_base = _score(original, matrices, config, source_local=0)
    hazard_score = _score(full_mht._edge_score, matrices, config, source_local=2)
    hazard_base = _score(original, matrices, config, source_local=2)

    anchor_delta = anchor_score - anchor_base
    hazard_delta = hazard_score - hazard_base

    assert anchor_delta > 0.0
    assert hazard_delta < 0.0
    assert anchor_delta > hazard_delta


def test_prior_survival_selected_edge_summary_exposes_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cell_probabilities(monkeypatch)
    matrices = _prior_matrix()
    install_full_mht_prior_survival_scoring()
    config = full_mht.FullMHTConfig(track2p_prior_weight=0.0)
    object.__setattr__(config, "track2p_prior_survival_weight", 1.0)

    summary = full_mht._selected_edge_summary(
        (object(), object()),
        matrices,
        active_source=full_mht._ActiveTrackSource(
            row_index=2,
            source_session=0,
            source_roi=30,
            gap_length=0,
        ),
        target_session=1,
        target_roi=31,
        config=config,
        track2p_prior_edges=_prior_edges(),
    )

    assert summary["track2p_prior_survival_score"] < 0.0
    assert summary["track2p_prior_survival_weighted_score"] < 0.0
    assert "|survival=" in summary["summary"]
    assert "|survival_weighted=" in summary["summary"]


def test_prior_survival_scoring_is_disabled_without_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cell_probabilities(monkeypatch)
    matrices = _prior_matrix()
    install_full_mht_prior_survival_scoring()
    original = _original_edge_score()
    config = full_mht.FullMHTConfig(track2p_prior_weight=0.0)

    patched_score = _score(full_mht._edge_score, matrices, config, source_local=0)
    original_score = _score(original, matrices, config, source_local=0)

    assert patched_score == pytest.approx(original_score)


def test_prior_survival_scoring_falls_back_without_pseudo_class_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cell_probabilities(monkeypatch)
    matrices = _prior_matrix()
    install_full_mht_prior_survival_scoring()
    original = _original_edge_score()
    config = full_mht.FullMHTConfig(track2p_prior_weight=0.0)
    object.__setattr__(config, "track2p_prior_survival_weight", 3.0)
    object.__setattr__(config, "track2p_prior_survival_min_examples_per_class", 3)

    patched_score = _score(full_mht._edge_score, matrices, config, source_local=2)
    original_score = _score(original, matrices, config, source_local=2)

    assert patched_score == pytest.approx(original_score)


def test_prior_survival_scoring_installer_is_idempotent() -> None:
    install_full_mht_prior_survival_scoring()
    first = full_mht._edge_score
    original = _original_edge_score()

    install_full_mht_prior_survival_scoring()

    assert full_mht._edge_score is first
    assert _original_edge_score() is original
