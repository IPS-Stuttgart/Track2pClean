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
        "track2p-supported-gap-cleanup",
    ]
    assert all(preset.config.method == "global-assignment" for preset in presets)
    assert all(preset.config.reference_kind == "manual-gt" for preset in presets)
    assert all(preset.config.include_non_cells for preset in presets[:3])
    assert all(preset.config.weighted_masks for preset in presets[:3])

    shifted, pruned, consensus, supported_gap = presets
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
    assert supported_gap.runner == "supported-gap-cleanup"
    assert supported_gap.config.transform_type == "affine"
    assert supported_gap.config.include_non_cells is False
    assert supported_gap.config.weighted_masks is False
    assert supported_gap.runner_kwargs is not None
    assert supported_gap.runner_kwargs["min_bridge_support"] == 1
    assert supported_gap.runner_kwargs["reject_conflicting_bridge_support"] is True


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
    assert rows[3]["runner"] == "supported-gap-cleanup"
    assert rows[3]["supported_gap_cleanup"] is True
