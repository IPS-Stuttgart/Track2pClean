from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from bayescatrack.accuracy_presets import (
    AccuracyPreset,
    accuracy_preset_metadata,
    build_track2p_accuracy_presets,
    run_track2p_accuracy_presets,
)
from bayescatrack.experiments.track2p_benchmark import SubjectBenchmarkResult


class _MalformedIndex:
    def __index__(self) -> int:
        raise ValueError("bad integer adapter")


class _OverflowingIndex:
    def __index__(self) -> int:
        raise OverflowError("integer adapter overflow")


class _OverflowingFloat:
    def __float__(self) -> float:
        raise OverflowError("float adapter overflow")


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
        "track2p-stability-cleanup",
        "track2p-supported-gap-cleanup",
        "track2p-confidence-ordered-strict-gap-cleanup",
        "track2p-teacher-adjacent-rescue",
        "track2p-teacher-fn-rescue",
        "track2p-residual-union-action-specific-rescue",
        "track2p-completing-rescue-action-specific",
    ]
    assert all(preset.config.method == "global-assignment" for preset in presets)
    assert all(preset.config.reference_kind == "manual-gt" for preset in presets)
    assert all(preset.config.include_non_cells for preset in presets[:3])
    assert all(preset.config.weighted_masks for preset in presets[:3])

    (
        shifted,
        pruned,
        consensus,
        stability,
        supported_gap,
        confidence_gap,
        teacher,
        teacher_fn,
        teacher_residual_union,
        teacher_completing_rescue,
    ) = presets
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
    assert stability.runner == "stability-cleanup"
    assert stability.config.transform_type == "affine"
    assert stability.config.max_gap == 1
    assert stability.config.include_non_cells is False
    assert stability.config.weighted_masks is False
    assert stability.runner_kwargs is not None
    assert stability.runner_kwargs["threshold_method"] == "min"
    cleanup_kwargs = stability.runner_kwargs["cleanup_config_kwargs"]
    assert isinstance(cleanup_kwargs, dict)
    assert cleanup_kwargs["base_iou_distance_threshold"] == 12.0
    assert cleanup_kwargs["min_support_fraction"] == 2.0 / 3.0
    assert cleanup_kwargs["min_side_observations"] == 2
    assert supported_gap.runner == "supported-gap-cleanup"
    assert supported_gap.config.transform_type == "affine"
    assert supported_gap.config.include_non_cells is False
    assert supported_gap.config.weighted_masks is False
    assert supported_gap.runner_kwargs is not None
    assert supported_gap.runner_kwargs["min_bridge_support"] == 1
    assert supported_gap.runner_kwargs["reject_conflicting_bridge_support"] is True
    assert confidence_gap.runner == "confidence-ordered-strict-gap-cleanup"
    assert confidence_gap.config is supported_gap.config
    assert confidence_gap.runner_kwargs is not None
    assert confidence_gap.runner_kwargs["threshold_method"] == "min"
    assert confidence_gap.runner_kwargs["iou_distance_threshold"] == 12.0
    confidence_cleanup_kwargs = confidence_gap.runner_kwargs["cleanup_config_kwargs"]
    assert isinstance(confidence_cleanup_kwargs, dict)
    assert confidence_cleanup_kwargs["require_complete_track"] is True
    assert teacher.runner == "teacher-adjacent-rescue"
    assert teacher.config is stability.config
    assert teacher.runner_kwargs is not None
    assert teacher.runner_kwargs["threshold_method"] == "min"
    assert teacher.runner_kwargs["iou_distance_threshold"] == 12.0
    assert teacher.runner_kwargs["allow_completing_rescue"] is False
    teacher_cleanup_kwargs = teacher.runner_kwargs["cleanup_config_kwargs"]
    assert isinstance(teacher_cleanup_kwargs, dict)
    assert teacher_cleanup_kwargs["require_complete_track"] is True
    assert teacher_fn.runner == "teacher-adjacent-rescue"
    assert teacher_fn.config is stability.config
    assert teacher_fn.runner_kwargs is not None
    assert (
        teacher_fn.runner_kwargs["teacher_repair_preset"]
        == "track2p-fn-high-confidence"
    )
    assert teacher_fn.runner_kwargs["threshold_method"] == "min"
    teacher_fn_cleanup_kwargs = teacher_fn.runner_kwargs["cleanup_config_kwargs"]
    assert isinstance(teacher_fn_cleanup_kwargs, dict)
    assert teacher_residual_union.runner == "teacher-adjacent-rescue"
    assert teacher_residual_union.config is stability.config
    assert teacher_residual_union.runner_kwargs is not None
    assert (
        teacher_residual_union.runner_kwargs["teacher_repair_preset"]
        == "residual-union-action-specific"
    )
    assert teacher_residual_union.runner_kwargs["threshold_method"] == "min"
    residual_union_cleanup_kwargs = teacher_residual_union.runner_kwargs[
        "cleanup_config_kwargs"
    ]
    assert isinstance(residual_union_cleanup_kwargs, dict)
    assert teacher_completing_rescue.runner == "teacher-adjacent-rescue"
    assert teacher_completing_rescue.config is stability.config
    assert teacher_completing_rescue.runner_kwargs is not None
    assert (
        teacher_completing_rescue.runner_kwargs["teacher_repair_preset"]
        == "completing-rescue-action-specific"
    )
    assert teacher_completing_rescue.runner_kwargs["threshold_method"] == "min"
    completing_rescue_cleanup_kwargs = teacher_completing_rescue.runner_kwargs[
        "cleanup_config_kwargs"
    ]
    assert isinstance(completing_rescue_cleanup_kwargs, dict)


@pytest.mark.parametrize(
    "max_gap",
    [_MalformedIndex(), _OverflowingIndex(), _OverflowingFloat()],
)
def test_build_track2p_accuracy_presets_normalizes_invalid_max_gap_adapters(
    max_gap: object,
) -> None:
    with pytest.raises(ValueError, match="max_gap must be a positive integer"):
        build_track2p_accuracy_presets(
            "/data/track2p",
            max_gap=max_gap,  # type: ignore[arg-type]
            progress=False,
        )


def test_build_track2p_accuracy_presets_normalizes_invalid_cost_threshold_adapter() -> (
    None
):
    with pytest.raises(
        ValueError, match="cost_threshold must be a finite non-negative value or None"
    ):
        build_track2p_accuracy_presets(
            "/data/track2p",
            cost_threshold=_OverflowingFloat(),  # type: ignore[arg-type]
            progress=False,
        )


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
    assert rows[3]["runner"] == "stability-cleanup"
    assert rows[3]["stability_cleanup"] is True
    assert rows[4]["runner"] == "supported-gap-cleanup"
    assert rows[4]["supported_gap_cleanup"] is True
    assert rows[4]["confidence_ordered_strict_gap_cleanup"] is False
    assert rows[4]["teacher_adjacent_rescue"] is False
    assert rows[5]["runner"] == "confidence-ordered-strict-gap-cleanup"
    assert rows[5]["supported_gap_cleanup"] is False
    assert rows[5]["confidence_ordered_strict_gap_cleanup"] is True
    assert rows[5]["teacher_adjacent_rescue"] is False
    assert rows[6]["runner"] == "teacher-adjacent-rescue"
    assert rows[6]["teacher_adjacent_rescue"] is True
    assert rows[7]["runner"] == "teacher-adjacent-rescue"
    assert rows[7]["teacher_adjacent_rescue"] is True
    assert rows[8]["runner"] == "teacher-adjacent-rescue"
    assert rows[8]["teacher_adjacent_rescue"] is True


def test_confidence_strict_gap_preset_runner_builds_typed_configs(monkeypatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    fake_result = SubjectBenchmarkResult(
        subject="jm_synthetic",
        variant="confidence strict gap",
        method="track2p-policy-confidence-ordered-strict-gated-gap-cleanup",
        scores={},
        n_sessions=2,
        reference_source="manual_gt",
    )

    class FakeComponentCleanupConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeStrictGapGateConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def fake_run(config: object, **kwargs: object) -> SimpleNamespace:
        calls.append((config, dict(kwargs)))
        return SimpleNamespace(results=(fake_result,))

    from bayescatrack import accuracy_presets as module

    monkeypatch.setattr(
        module,
        "build_track2p_accuracy_presets",
        lambda *args, **kwargs: (
            AccuracyPreset(
                name="track2p-confidence-ordered-strict-gap-cleanup",
                description="synthetic confidence strict gap preset",
                config=build_track2p_accuracy_presets("/unused", progress=False)[
                    -3
                ].config,
                runner="confidence-ordered-strict-gap-cleanup",
                runner_kwargs={
                    "threshold_method": "min",
                    "cleanup_config_kwargs": {"split_risk_threshold": 1.25},
                    "gate_config_kwargs": {"min_threshold_margin": 0.35},
                },
            ),
        ),
    )
    fake_module = SimpleNamespace(
        ComponentCleanupConfig=FakeComponentCleanupConfig,
        StrictGapGateConfig=FakeStrictGapGateConfig,
        run_track2p_policy_confidence_ordered_strict_gated_gap_cleanup=fake_run,
    )
    import bayescatrack.experiments

    monkeypatch.setattr(
        bayescatrack.experiments,
        "track2p_policy_confidence_ordered_strict_gap_cleanup",
        fake_module,
        raising=False,
    )

    output = run_track2p_accuracy_presets(
        "/data/track2p",
        preset_names=cast(object, ("track2p-confidence-ordered-strict-gap-cleanup",)),
    )

    assert output == {"track2p-confidence-ordered-strict-gap-cleanup": [fake_result]}
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["threshold_method"] == "min"
    assert isinstance(kwargs["cleanup_config"], FakeComponentCleanupConfig)
    assert kwargs["cleanup_config"].kwargs == {"split_risk_threshold": 1.25}
    assert isinstance(kwargs["gate_config"], FakeStrictGapGateConfig)
    assert kwargs["gate_config"].kwargs == {"min_threshold_margin": 0.35}


def test_teacher_adjacent_rescue_preset_runner_builds_typed_config(monkeypatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    fake_result = SubjectBenchmarkResult(
        subject="jm_synthetic",
        variant="teacher adjacent rescue",
        method="track2p-policy-teacher-adjacent-rescue",
        scores={},
        n_sessions=2,
        reference_source="manual_gt",
    )

    class FakeComponentCleanupConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def fake_run(config: object, **kwargs: object) -> SimpleNamespace:
        calls.append((config, dict(kwargs)))
        return SimpleNamespace(results=(fake_result,))

    from bayescatrack import accuracy_presets as module

    monkeypatch.setattr(
        module,
        "build_track2p_accuracy_presets",
        lambda *args, **kwargs: (
            AccuracyPreset(
                name="track2p-teacher-adjacent-rescue",
                description="synthetic teacher rescue preset",
                config=build_track2p_accuracy_presets("/unused", progress=False)[
                    -1
                ].config,
                runner="teacher-adjacent-rescue",
                runner_kwargs={
                    "threshold_method": "min",
                    "cleanup_config_kwargs": {"split_risk_threshold": 1.25},
                    "allow_completing_rescue": False,
                },
            ),
        ),
    )
    fake_module = SimpleNamespace(
        ComponentCleanupConfig=FakeComponentCleanupConfig,
        run_track2p_policy_teacher_adjacent_rescue=fake_run,
    )
    import bayescatrack.experiments

    monkeypatch.setattr(
        bayescatrack.experiments,
        "track2p_policy_teacher_adjacent_rescue",
        fake_module,
        raising=False,
    )

    output = run_track2p_accuracy_presets(
        "/data/track2p",
        preset_names=cast(object, ("track2p-teacher-adjacent-rescue",)),
    )

    assert output == {"track2p-teacher-adjacent-rescue": [fake_result]}
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["threshold_method"] == "min"
    assert kwargs["allow_completing_rescue"] is False
    assert isinstance(kwargs["cleanup_config"], FakeComponentCleanupConfig)
    assert kwargs["cleanup_config"].kwargs == {"split_risk_threshold": 1.25}
