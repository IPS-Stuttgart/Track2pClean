from __future__ import annotations

from pathlib import Path

from bayescatrack.accuracy_presets import (
    accuracy_preset_metadata,
    build_track2p_accuracy_presets,
)


def test_build_track2p_accuracy_presets_exposes_stronger_structural_configs() -> None:
    presets = build_track2p_accuracy_presets(
        Path("/data/track2p"),
        reference=Path("/data/manual_gt"),
        progress=False,
    )

    assert [preset.name for preset in presets] == [
        "registered-shifted-iou-safe",
        "roi-aware-shifted-pruned",
        "roi-aware-shifted-consensus",
    ]
    assert all(preset.config.method == "global-assignment" for preset in presets)
    assert all(preset.config.reference_kind == "manual-gt" for preset in presets)
    assert all(preset.config.include_non_cells for preset in presets)
    assert all(preset.config.weighted_masks for preset in presets)

    shifted, pruned, consensus = presets
    assert shifted.config.cost == "registered-shifted-iou"
    assert shifted.config.higher_order_consistency_config is not None
    assert pruned.config.cost == "roi-aware-shifted"
    assert pruned.config.candidate_pruning_config == {
        "row_top_k": 24,
        "column_top_k": 24,
        "max_cost": 6.0,
    }
    assert pruned.config.dynamic_edge_prior_config is not None
    assert pruned.config.activity_tie_breaker_weight > 0.0
    assert consensus.config.consensus_prior_config is not None
    assert consensus.config.consensus_prior_config["min_votes"] == 2


def test_accuracy_preset_metadata_is_compact_and_serializable() -> None:
    presets = build_track2p_accuracy_presets(
        "/data/track2p",
        cost_threshold=None,
        progress=False,
    )
    rows = accuracy_preset_metadata(presets)

    assert rows[0]["name"] == "registered-shifted-iou-safe"
    assert rows[0]["cost_threshold"] == "none"
    assert rows[1]["candidate_pruning"] is True
    assert rows[1]["dynamic_edge_prior"] is True
    assert rows[2]["consensus_prior"] is True
