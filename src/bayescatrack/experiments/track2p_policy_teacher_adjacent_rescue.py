"""Track2p-teacher adjacent rescue after component cleanup.

This is a deliberately narrow Track2p-teacher hybrid ablation.  It starts from
the frozen Track2pPolicy component-cleanup prediction and admits only adjacent
Track2p teacher edges that can extend an already existing seed-anchored
component without creating duplicate source/target observations.  Teacher edges
can also backfill a missing source observation or merge two compatible
seed-anchored fragments when the edit has no ROI conflicts.  Completing a row by
isolated insertion remains disabled unless explicitly allowed or unless the
completed row is also present as a complete Track2p teacher row, while
complete-row fragment merges can be enabled separately for a narrower
stitch-only rescue. A seed-source backfill opt-in can now be used independently
of broad source backfill, allowing a narrow Track2p-supported probe of the
missing seed-session ROI residual bucket without also admitting arbitrary
non-seed source insertions. A completing-rescue action filter can spend a tiny
teacher edit budget only on edges that would complete a predicted row, which is
useful after residual audits show that non-completing inserts are often safe but
metric-neutral. Missing-seed repair presets also restrict the
teacher action family to seed-source backfills by default, so a tiny edit budget
cannot be spent on unrelated target extensions before the seed-source hypothesis
is tested. Residual-union presets additionally use the cell-priority dynamic
ordering: once the residual FN / missing-seed action gate has selected a tiny
edit budget, high endpoint cell probability and shape support are stronger
tie-breakers than raw registered IoU. The command does not use manual GT labels
to choose edges.

Residual-union runs can also use per-action edit caps.  Those caps keep a tiny
teacher budget from being consumed entirely by one residual family, making it
possible to test a balanced row that spends one slot on the missing-seed bucket
and the remaining slots on Track2p-supported target-extension FNs.

Action-specific feature gates allow a residual-union preset to be strict for
ordinary target extensions while keeping a separate, seed-source-specific gate
for the missing seed-session ROI bucket.

The ``completing-seed-source-backfill`` action filter is narrower still: it
spends a tiny teacher-edit budget only on seed-session source backfills that
would immediately complete the affected predicted row.  This directly targets
the residual audit's missing-seed complete-FN bucket without admitting ordinary
target extensions, non-seed backfills, or non-completing seed insertions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _predict_subject_tracks,
    _reference_matrix,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_audit import TrackEdge, track_edge_counter
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentAuditOutput,
    ComponentCleanupConfig,
    _evaluated_prediction_rows,
    _mark_applied_splits,
    _normalize_int_track_matrix,
    apply_weakest_bridge_splits,
    component_audit_rows,
)
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    ResidualFeature,
    _feature_subset_for_edges,
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_METHOD = "track2p-policy-teacher-adjacent-rescue"
TeacherEdgeOrder = Literal[
    "lexicographic",
    "structural",
    "dynamic-structural",
    "confidence",
    "cell-confidence",
    "dynamic-confidence",
    "dynamic-cell-confidence",
    "dynamic-seed-confidence",
    "dynamic-seed-cell-confidence",
]
TeacherActionFilter = Literal[
    "all",
    "target-extension",
    "source-backfill",
    "seed-source-backfill",
    "fragment-merge",
    "target-extension-or-seed-source-backfill",
    "completing-rescue",
    "completing-seed-source-backfill",
]
TeacherFeaturePreset = Literal[
    "none",
    "local-support",
    "high-confidence",
    "cell-high-confidence",
    "cell-confident",
    "track2p-fn-rescue",
    "residual-fn",
    "residual-fn-cell-confident",
    "moderate-iou-cell-confidence",
    "seed-source-high-confidence",
    "seed-source-cell-confident",
    "seed-source-moderate-iou",
]
TeacherRepairPreset = Literal[
    "none",
    "missing-seed-high-confidence",
    "missing-seed-cell-confident",
    "missing-seed-moderate-iou",
    "missing-seed-completing-cell-confident",
    "missing-seed-completing-moderate-iou",
    "track2p-fn-high-confidence",
    "track2p-fn-moderate-iou-cell-confident",
    "track2p-fn-moderate-iou-cell-confidence",
    "residual-union-cell-confident",
    "residual-union-action-specific",
    "residual-union-action-balanced",
    "residual-union-balanced",
    "completing-rescue-action-specific",
    "complete-row-rescue-action-specific",
]


def _integral_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    raise ValueError(f"{name} must be an integer")


def _positive_int_value(value: Any, *, name: str) -> int:
    numeric = _integral_value(value, name=name)
    if numeric <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return numeric


def _nonnegative_int_value(value: Any, *, name: str) -> int:
    numeric = _integral_value(value, name=name)
    if numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return numeric


def _bool_value(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _optional_bool_value(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    return _bool_value(value, name=name)


def _positive_int_arg(value: str) -> int:
    try:
        return _positive_int_value(value, name="value")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _nonnegative_int_arg(value: str) -> int:
    try:
        return _nonnegative_int_value(value, name="value")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


@dataclass(frozen=True)
class TeacherAdjacentRescueReport:
    """Prediction plus teacher-rescue diagnostic rows."""

    tracks: np.ndarray
    rows: tuple[dict[str, int | str], ...]


@dataclass(frozen=True, init=False)
class TeacherEdgeFeatureGate:
    """Label-free local-evidence gate for Track2p teacher rescue edges."""

    min_registered_iou: float | None
    max_registered_iou: float | None
    min_threshold_margin: float | None
    min_row_margin: float | None
    min_column_margin: float | None
    max_centroid_distance: float | None
    min_area_ratio: float | None
    min_cell_probability: float | None
    require_hungarian: bool

    def __init__(
        self,
        *,
        min_registered_iou: float | None = None,
        max_registered_iou: float | None = None,
        min_threshold_margin: float | None = None,
        min_row_margin: float | None = None,
        min_column_margin: float | None = None,
        max_centroid_distance: float | None = None,
        min_area_ratio: float | None = None,
        min_cell_probability: float | None = None,
        require_hungarian: bool = False,
        require_hungarian_assignment: bool | None = None,
        require_assigned_by_hungarian: bool | None = None,
    ) -> None:
        if require_hungarian_assignment is not None:
            require_hungarian = _bool_value(
                require_hungarian_assignment, name="require_hungarian_assignment"
            )
        if require_assigned_by_hungarian is not None:
            require_hungarian = _bool_value(
                require_assigned_by_hungarian, name="require_assigned_by_hungarian"
            )
        require_hungarian = _bool_value(
            require_hungarian, name="require_hungarian"
        )
        object.__setattr__(self, "min_registered_iou", min_registered_iou)
        object.__setattr__(self, "max_registered_iou", max_registered_iou)
        object.__setattr__(self, "min_threshold_margin", min_threshold_margin)
        object.__setattr__(self, "min_row_margin", min_row_margin)
        object.__setattr__(self, "min_column_margin", min_column_margin)
        object.__setattr__(self, "max_centroid_distance", max_centroid_distance)
        object.__setattr__(self, "min_area_ratio", min_area_ratio)
        object.__setattr__(self, "min_cell_probability", min_cell_probability)
        object.__setattr__(self, "require_hungarian", require_hungarian)

    @property
    def require_hungarian_assignment(self) -> bool:
        return self.require_hungarian

    @property
    def require_assigned_by_hungarian(self) -> bool:
        return self.require_hungarian

    @property
    def enabled(self) -> bool:
        return bool(
            self.require_hungarian
            or self.min_registered_iou is not None
            or self.max_registered_iou is not None
            or self.min_threshold_margin is not None
            or self.min_row_margin is not None
            or self.min_column_margin is not None
            or self.max_centroid_distance is not None
            or self.min_area_ratio is not None
            or self.min_cell_probability is not None
        )


TeacherFeatureGate = TeacherEdgeFeatureGate


def _teacher_edge_order_requires_feature_index(
    edge_order: TeacherEdgeOrder | str,
) -> bool:
    """Return whether a teacher-edge order needs local feature extraction."""

    return str(edge_order) in {
        "confidence",
        "cell-confidence",
        "dynamic-confidence",
        "dynamic-cell-confidence",
        "dynamic-seed-confidence",
        "dynamic-seed-cell-confidence",
    }


def _teacher_edge_order_uses_confidence_features(
    edge_order: TeacherEdgeOrder | str,
) -> bool:
    """Compatibility alias for confidence-feature edge-order detection."""

    return _teacher_edge_order_requires_feature_index(edge_order)


def teacher_feature_gate_from_preset(
    preset: TeacherFeaturePreset | str,
) -> TeacherEdgeFeatureGate | None:
    """Return a label-free teacher-rescue feature gate preset."""

    normalized = str(preset).strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized == "local-support":
        return TeacherEdgeFeatureGate(
            min_threshold_margin=0.0,
            min_row_margin=0.0,
            min_column_margin=0.0,
            max_centroid_distance=6.0,
            min_area_ratio=0.60,
            require_hungarian=True,
        )
    if normalized == "high-confidence":
        return TeacherEdgeFeatureGate(
            min_registered_iou=0.20,
            min_threshold_margin=0.05,
            min_row_margin=0.0,
            min_column_margin=0.0,
            max_centroid_distance=4.0,
            min_area_ratio=0.70,
            require_hungarian=True,
        )
    if normalized in {"cell-high-confidence", "cell-confident", "cell-confidence"}:
        return TeacherEdgeFeatureGate(
            min_registered_iou=0.20,
            min_threshold_margin=0.05,
            min_row_margin=0.0,
            min_column_margin=0.0,
            max_centroid_distance=4.0,
            min_area_ratio=0.70,
            min_cell_probability=0.80,
            require_hungarian=True,
        )
    if normalized in {"residual-fn", "residual-fn-rescue", "teacher-fn"}:
        return TeacherEdgeFeatureGate(
            min_registered_iou=0.10,
            max_centroid_distance=6.0,
            min_area_ratio=0.45,
            min_cell_probability=0.50,
            require_hungarian=False,
        )
    if normalized in {
        "residual-fn-cell-confident",
        "residual-fn-cell-confidence",
        "teacher-fn-cell-confident",
    }:
        return TeacherEdgeFeatureGate(
            min_registered_iou=0.10,
            max_centroid_distance=6.0,
            min_area_ratio=0.45,
            min_cell_probability=0.80,
            require_hungarian=False,
        )
    if normalized == "track2p-fn-rescue":
        return TeacherEdgeFeatureGate(
            min_registered_iou=0.10,
            max_centroid_distance=6.0,
            min_area_ratio=0.45,
            min_cell_probability=0.60,
            require_hungarian=False,
        )
    if normalized in {
        "moderate-iou-cell-confidence",
        "moderate-iou-cell-confident",
    }:
        return TeacherEdgeFeatureGate(
            min_registered_iou=0.10,
            max_registered_iou=0.55,
            max_centroid_distance=6.0,
            min_area_ratio=0.60,
            min_cell_probability=0.80,
            require_hungarian=False,
        )
    if normalized == "seed-source-high-confidence":
        return TeacherEdgeFeatureGate(
            min_registered_iou=0.15,
            max_centroid_distance=6.0,
            min_area_ratio=0.60,
            min_cell_probability=0.70,
            require_hungarian=False,
        )
    if normalized in {"seed-source-cell-confident", "missing-seed-cell-confident"}:
        return TeacherEdgeFeatureGate(
            min_registered_iou=0.0,
            max_centroid_distance=6.0,
            min_area_ratio=0.60,
            min_cell_probability=0.85,
            require_hungarian=False,
        )
    if normalized in {"seed-source-moderate-iou", "missing-seed-moderate-iou"}:
        return TeacherEdgeFeatureGate(
            min_registered_iou=0.10,
            max_registered_iou=0.55,
            max_centroid_distance=6.0,
            min_area_ratio=0.60,
            min_cell_probability=0.80,
            require_hungarian=False,
        )
    raise ValueError(f"Unsupported teacher feature preset: {preset!r}")


def teacher_adjacent_repair_preset_kwargs(
    preset: TeacherRepairPreset | str,
) -> dict[str, Any]:
    """Return label-free macro options for narrow teacher-rescue repair modes."""

    normalized = str(preset).strip().lower()
    if normalized in {"", "none"}:
        return {}
    if normalized == "missing-seed-high-confidence":
        return {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-seed-confidence",
            "teacher_action_filter": "seed-source-backfill",
            "teacher_feature_preset": "seed-source-high-confidence",
            "min_component_observations": 2,
            "max_applied_edits": 2,
        }
    if normalized == "missing-seed-cell-confident":
        return {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-seed-confidence",
            "teacher_action_filter": "seed-source-backfill",
            "teacher_feature_preset": "seed-source-cell-confident",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        }
    if normalized == "missing-seed-moderate-iou":
        return {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-seed-confidence",
            "teacher_action_filter": "seed-source-backfill",
            "teacher_feature_preset": "seed-source-moderate-iou",
            "min_component_observations": 2,
            "max_applied_edits": 2,
        }
    if normalized == "missing-seed-completing-cell-confident":
        return {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "allow_fragment_merges": False,
            "teacher_edge_order": "dynamic-seed-cell-confidence",
            "teacher_action_filter": "completing-seed-source-backfill",
            "teacher_feature_preset": "seed-source-cell-confident",
            "min_component_observations": 2,
            "max_applied_edits": 1,
        }
    if normalized == "missing-seed-completing-moderate-iou":
        return {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "allow_fragment_merges": False,
            "teacher_edge_order": "dynamic-seed-cell-confidence",
            "teacher_action_filter": "completing-seed-source-backfill",
            "teacher_feature_preset": "seed-source-moderate-iou",
            "min_component_observations": 2,
            "max_applied_edits": 1,
        }
    if normalized in {"track2p-fn-high-confidence", "track2p-fn-target-extension"}:
        return {
            "teacher_action_filter": "target-extension",
            "teacher_edge_order": "dynamic-confidence",
            "teacher_feature_preset": "track2p-fn-rescue",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        }
    if normalized in {
        "track2p-fn-moderate-iou-cell-confident",
        "track2p-fn-moderate-iou-cell-confidence",
    }:
        return {
            "teacher_action_filter": "target-extension",
            "teacher_edge_order": "dynamic-cell-confidence",
            "teacher_feature_preset": "moderate-iou-cell-confidence",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        }
    if normalized in {
        "residual-union-cell-confident",
        "residual-union",
        "track2p-fn-or-missing-seed",
    }:
        return {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "allow_fragment_merges": False,
            "teacher_action_filter": "target-extension-or-seed-source-backfill",
            "teacher_edge_order": "dynamic-seed-cell-confidence",
            "teacher_feature_preset": "residual-fn-cell-confident",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        }
    if normalized == "residual-union-action-specific":
        return {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "allow_fragment_merges": False,
            "teacher_action_filter": "target-extension-or-seed-source-backfill",
            "teacher_edge_order": "dynamic-seed-cell-confidence",
            "teacher_feature_preset": "none",
            "target_extension_feature_preset": "moderate-iou-cell-confidence",
            "seed_source_feature_preset": "seed-source-cell-confident",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        }
    if normalized in {"residual-union-action-balanced", "residual-union-balanced"}:
        return {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "allow_fragment_merges": False,
            "teacher_action_filter": "target-extension-or-seed-source-backfill",
            "teacher_edge_order": "dynamic-seed-cell-confidence",
            "teacher_feature_preset": "none",
            "target_extension_feature_preset": "moderate-iou-cell-confidence",
            "seed_source_feature_preset": "seed-source-cell-confident",
            "min_component_observations": 2,
            "max_applied_edits": 3,
            # Reserve one slot for the missing-seed bucket while still allowing
            # the residual Track2p-FN target-extension bucket to claim two edits.
            "max_seed_source_backfill_edits": 1,
            "max_target_extension_edits": 2,
        }
    if normalized in {
        "completing-rescue-action-specific",
        "complete-row-rescue-action-specific",
        "complete-row-action-specific",
    }:
        return {
            "allow_teacher_complete_row_rescue": True,
            "allow_completing_rescue": True,
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "allow_fragment_merges": False,
            "teacher_action_filter": "completing-rescue",
            "teacher_edge_order": "dynamic-seed-cell-confidence",
            "teacher_feature_preset": "none",
            "target_extension_feature_preset": "moderate-iou-cell-confidence",
            "seed_source_feature_preset": "seed-source-cell-confident",
            "min_component_observations": 2,
            "max_applied_edits": 2,
        }
    raise ValueError(f"Unsupported teacher repair preset: {preset!r}")


def merge_teacher_feature_gates(
    preset_gate: TeacherEdgeFeatureGate | None,
    manual_gate: TeacherEdgeFeatureGate,
) -> TeacherEdgeFeatureGate | None:
    """Merge a preset teacher gate with explicit CLI threshold overrides."""

    if preset_gate is None:
        return manual_gate if manual_gate.enabled else None
    if not manual_gate.enabled:
        return preset_gate
    return TeacherEdgeFeatureGate(
        min_registered_iou=(
            manual_gate.min_registered_iou
            if manual_gate.min_registered_iou is not None
            else preset_gate.min_registered_iou
        ),
        max_registered_iou=(
            manual_gate.max_registered_iou
            if manual_gate.max_registered_iou is not None
            else preset_gate.max_registered_iou
        ),
        min_threshold_margin=(
            manual_gate.min_threshold_margin
            if manual_gate.min_threshold_margin is not None
            else preset_gate.min_threshold_margin
        ),
        min_row_margin=(
            manual_gate.min_row_margin
            if manual_gate.min_row_margin is not None
            else preset_gate.min_row_margin
        ),
        min_column_margin=(
            manual_gate.min_column_margin
            if manual_gate.min_column_margin is not None
            else preset_gate.min_column_margin
        ),
        max_centroid_distance=(
            manual_gate.max_centroid_distance
            if manual_gate.max_centroid_distance is not None
            else preset_gate.max_centroid_distance
        ),
        min_area_ratio=(
            manual_gate.min_area_ratio
            if manual_gate.min_area_ratio is not None
            else preset_gate.min_area_ratio
        ),
        min_cell_probability=(
            manual_gate.min_cell_probability
            if manual_gate.min_cell_probability is not None
            else preset_gate.min_cell_probability
        ),
        require_hungarian=manual_gate.require_hungarian
        or preset_gate.require_hungarian,
    )


def _resolve_source_backfill_alias(
    allow_source_backfill: bool,
    allow_source_inserts: bool | None,
    allow_source_insertions: bool | None = None,
) -> bool:
    if allow_source_insertions is not None:
        return _bool_value(
            allow_source_insertions, name="allow_source_insertions"
        )
    if allow_source_inserts is None:
        return _bool_value(allow_source_backfill, name="allow_source_backfill")
    return _bool_value(allow_source_inserts, name="allow_source_inserts")


def _source_backfill_allowed_for_edge(
    source_backfill_enabled: bool,
    *,
    allow_seed_source_backfill: bool,
    session_a: int,
    seed_session: int,
) -> bool:
    """Return whether this missing-source edge may be considered for backfill."""

    return _bool_value(
        source_backfill_enabled, name="source_backfill_enabled"
    ) or (
        _bool_value(
            allow_seed_source_backfill, name="allow_seed_source_backfill"
        )
        and int(session_a) == int(seed_session)
    )


def _teacher_completion_gate_kwargs(
    *,
    allow_teacher_complete_row_rescue: bool,
    allow_teacher_supported_completion: bool,
    allow_teacher_supported_completing_rescue: bool,
    allow_teacher_confirmed_completing_rescue: bool,
) -> dict[str, bool]:
    """Return teacher-completion kwargs without broadening exact aliases."""

    return {
        "allow_teacher_complete_row_rescue": _bool_value(
            allow_teacher_complete_row_rescue,
            name="allow_teacher_complete_row_rescue",
        ),
        "allow_teacher_supported_completion": _bool_value(
            allow_teacher_supported_completion,
            name="allow_teacher_supported_completion",
        ),
        "allow_teacher_supported_completing_rescue": _bool_value(
            allow_teacher_supported_completing_rescue,
            name="allow_teacher_supported_completing_rescue",
        ),
        "allow_teacher_confirmed_completing_rescue": _bool_value(
            allow_teacher_confirmed_completing_rescue,
            name="allow_teacher_confirmed_completing_rescue",
        ),
    }


def run_track2p_policy_teacher_adjacent_rescue(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    allow_completing_rescue: bool = False,
    allow_teacher_complete_row_rescue: bool = False,
    allow_teacher_supported_completion: bool = False,
    allow_teacher_supported_completing_rescue: bool = False,
    allow_teacher_confirmed_completing_rescue: bool = False,
    allow_completing_source_backfill: bool = False,
    allow_completing_fragment_merge: bool = False,
    allow_completing_fragment_merges: bool = False,
    allow_source_backfill: bool = True,
    allow_source_inserts: bool | None = None,
    allow_source_insertions: bool | None = None,
    allow_seed_source_backfill: bool = False,
    allow_seed_completing_backfill: bool = False,
    allow_seed_completing_rescue: bool = False,
    allow_completing_seed_source_backfill: bool = False,
    allow_fragment_merges: bool = True,
    teacher_edge_order: TeacherEdgeOrder = "structural",
    teacher_action_filter: TeacherActionFilter = "all",
    min_component_observations: int = 1,
    max_applied_edits: int | None = None,
    max_target_extension_edits: int | None = None,
    max_source_backfill_edits: int | None = None,
    max_seed_source_backfill_edits: int | None = None,
    max_fragment_merge_edits: int | None = None,
    max_completing_rescue_edits: int | None = None,
    teacher_feature_gate: TeacherEdgeFeatureGate | None = None,
    edge_feature_gate: TeacherEdgeFeatureGate | None = None,
    teacher_repair_preset: str = "none",
    teacher_feature_preset: str = "none",
    target_extension_feature_preset: str = "none",
    seed_source_feature_preset: str = "none",
) -> ComponentAuditOutput:
    """Run component cleanup followed by adjacent Track2p teacher rescue."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    cleanup_config = cleanup_config or ComponentCleanupConfig()
    min_component_observations = _positive_int_value(
        min_component_observations, name="min_component_observations"
    )
    allow_completing_rescue = _bool_value(
        allow_completing_rescue, name="allow_completing_rescue"
    )
    allow_teacher_complete_row_rescue = _bool_value(
        allow_teacher_complete_row_rescue,
        name="allow_teacher_complete_row_rescue",
    )
    allow_teacher_supported_completion = _bool_value(
        allow_teacher_supported_completion,
        name="allow_teacher_supported_completion",
    )
    allow_teacher_supported_completing_rescue = _bool_value(
        allow_teacher_supported_completing_rescue,
        name="allow_teacher_supported_completing_rescue",
    )
    allow_teacher_confirmed_completing_rescue = _bool_value(
        allow_teacher_confirmed_completing_rescue,
        name="allow_teacher_confirmed_completing_rescue",
    )
    allow_completing_source_backfill = _bool_value(
        allow_completing_source_backfill,
        name="allow_completing_source_backfill",
    )
    allow_completing_fragment_merge = _bool_value(
        allow_completing_fragment_merge,
        name="allow_completing_fragment_merge",
    )
    allow_completing_fragment_merges = _bool_value(
        allow_completing_fragment_merges,
        name="allow_completing_fragment_merges",
    )
    allow_source_backfill = _bool_value(
        allow_source_backfill, name="allow_source_backfill"
    )
    allow_source_inserts = _optional_bool_value(
        allow_source_inserts, name="allow_source_inserts"
    )
    allow_source_insertions = _optional_bool_value(
        allow_source_insertions, name="allow_source_insertions"
    )
    allow_seed_source_backfill = _bool_value(
        allow_seed_source_backfill, name="allow_seed_source_backfill"
    )
    allow_seed_completing_backfill = _bool_value(
        allow_seed_completing_backfill,
        name="allow_seed_completing_backfill",
    )
    allow_seed_completing_rescue = _bool_value(
        allow_seed_completing_rescue, name="allow_seed_completing_rescue"
    )
    allow_completing_seed_source_backfill = _bool_value(
        allow_completing_seed_source_backfill,
        name="allow_completing_seed_source_backfill",
    )
    allow_fragment_merges = _bool_value(
        allow_fragment_merges, name="allow_fragment_merges"
    )
    repair_kwargs = teacher_adjacent_repair_preset_kwargs(teacher_repair_preset)
    if repair_kwargs:
        if (
            "allow_source_backfill" in repair_kwargs
            and allow_source_inserts is None
            and allow_source_insertions is None
        ):
            allow_source_backfill = bool(repair_kwargs["allow_source_backfill"])
        if "allow_fragment_merges" in repair_kwargs:
            allow_fragment_merges = bool(repair_kwargs["allow_fragment_merges"])
        allow_completing_rescue = bool(
            allow_completing_rescue
            or repair_kwargs.get("allow_completing_rescue", False)
        )
        allow_teacher_complete_row_rescue = bool(
            allow_teacher_complete_row_rescue
            or repair_kwargs.get("allow_teacher_complete_row_rescue", False)
        )
        allow_teacher_supported_completion = bool(
            allow_teacher_supported_completion
            or repair_kwargs.get("allow_teacher_supported_completion", False)
        )
        allow_teacher_supported_completing_rescue = bool(
            allow_teacher_supported_completing_rescue
            or repair_kwargs.get("allow_teacher_supported_completing_rescue", False)
        )
        allow_completing_source_backfill = bool(
            allow_completing_source_backfill
            or repair_kwargs.get("allow_completing_source_backfill", False)
        )
        allow_completing_fragment_merges = bool(
            allow_completing_fragment_merges
            or repair_kwargs.get("allow_completing_fragment_merges", False)
        )
        allow_teacher_confirmed_completing_rescue = bool(
            allow_teacher_confirmed_completing_rescue
            or repair_kwargs.get("allow_teacher_confirmed_completing_rescue", False)
        )
        allow_seed_source_backfill = bool(
            allow_seed_source_backfill
            or repair_kwargs.get("allow_seed_source_backfill", False)
        )
        allow_completing_seed_source_backfill = bool(
            allow_completing_seed_source_backfill
            or repair_kwargs.get("allow_completing_seed_source_backfill", False)
        )
        if teacher_action_filter == "all" and "teacher_action_filter" in repair_kwargs:
            teacher_action_filter = cast(
                TeacherActionFilter, repair_kwargs["teacher_action_filter"]
            )
        if teacher_edge_order == "structural" and "teacher_edge_order" in repair_kwargs:
            teacher_edge_order = cast(
                TeacherEdgeOrder, repair_kwargs["teacher_edge_order"]
            )
        if (
            teacher_feature_preset == "none"
            and "teacher_feature_preset" in repair_kwargs
        ):
            teacher_feature_preset = str(repair_kwargs["teacher_feature_preset"])
        if (
            target_extension_feature_preset == "none"
            and "target_extension_feature_preset" in repair_kwargs
        ):
            target_extension_feature_preset = str(
                repair_kwargs["target_extension_feature_preset"]
            )
        if (
            seed_source_feature_preset == "none"
            and "seed_source_feature_preset" in repair_kwargs
        ):
            seed_source_feature_preset = str(
                repair_kwargs["seed_source_feature_preset"]
            )
        if "min_component_observations" in repair_kwargs:
            min_component_observations = max(
                min_component_observations,
                _positive_int_value(
                    repair_kwargs["min_component_observations"],
                    name="min_component_observations",
                ),
            )
        if max_applied_edits is None and "max_applied_edits" in repair_kwargs:
            max_applied_edits = int(repair_kwargs["max_applied_edits"])
        if (
            max_target_extension_edits is None
            and "max_target_extension_edits" in repair_kwargs
        ):
            max_target_extension_edits = int(
                repair_kwargs["max_target_extension_edits"]
            )
        if (
            max_source_backfill_edits is None
            and "max_source_backfill_edits" in repair_kwargs
        ):
            max_source_backfill_edits = int(repair_kwargs["max_source_backfill_edits"])
        if (
            max_seed_source_backfill_edits is None
            and "max_seed_source_backfill_edits" in repair_kwargs
        ):
            max_seed_source_backfill_edits = int(
                repair_kwargs["max_seed_source_backfill_edits"]
            )
        if (
            max_fragment_merge_edits is None
            and "max_fragment_merge_edits" in repair_kwargs
        ):
            max_fragment_merge_edits = int(repair_kwargs["max_fragment_merge_edits"])
        if (
            max_completing_rescue_edits is None
            and "max_completing_rescue_edits" in repair_kwargs
        ):
            max_completing_rescue_edits = int(
                repair_kwargs["max_completing_rescue_edits"]
            )
    source_backfill_enabled = _resolve_source_backfill_alias(
        allow_source_backfill, allow_source_inserts, allow_source_insertions
    )
    teacher_completion_kwargs = _teacher_completion_gate_kwargs(
        allow_teacher_complete_row_rescue=allow_teacher_complete_row_rescue,
        allow_teacher_supported_completion=allow_teacher_supported_completion,
        allow_teacher_supported_completing_rescue=(
            allow_teacher_supported_completing_rescue
        ),
        allow_teacher_confirmed_completing_rescue=(
            allow_teacher_confirmed_completing_rescue
        ),
    )
    allow_teacher_completion_gate_enabled = any(teacher_completion_kwargs.values())
    allow_seed_completion = bool(
        allow_seed_completing_backfill
        or allow_seed_completing_rescue
        or allow_completing_seed_source_backfill
    )
    allow_fragment_completion = bool(
        allow_completing_fragment_merge or allow_completing_fragment_merges
    )
    teacher_feature_gate = _resolve_teacher_feature_gate(
        teacher_feature_gate, edge_feature_gate
    )
    teacher_feature_gate = merge_teacher_feature_gates(
        teacher_feature_gate_from_preset(teacher_feature_preset),
        teacher_feature_gate or TeacherEdgeFeatureGate(),
    )
    target_extension_feature_gate = teacher_feature_gate_from_preset(
        target_extension_feature_preset
    )
    seed_source_feature_gate = teacher_feature_gate_from_preset(
        seed_source_feature_preset
    )
    results: list[SubjectBenchmarkResult] = []
    rescue_rows: list[dict[str, int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy teacher adjacent rescue requires independent "
                "manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        base_full = _component_cleanup_prediction(
            sessions,
            reference_tracks,
            config=policy_config,
            cleanup_config=cleanup_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
        )
        teacher_full, _variant = _predict_subject_tracks(
            subject_dir, replace(policy_config, method="track2p-baseline")
        )
        edge_feature_index = (
            _feature_subset_for_edges(
                sessions,
                set(track_edge_counter(_normalize_int_track_matrix(teacher_full))),
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
            )
            if _teacher_edge_order_requires_feature_index(teacher_edge_order)
            or _teacher_feature_gate_enabled(teacher_feature_gate)
            or _teacher_feature_gate_enabled(target_extension_feature_gate)
            or _teacher_feature_gate_enabled(seed_source_feature_gate)
            else {}
        )
        rescue = apply_teacher_adjacent_rescue_edges(
            base_full,
            teacher_full,
            seed_session=policy_config.seed_session,
            allow_completing_rescue=allow_completing_rescue,
            **teacher_completion_kwargs,
            allow_completing_source_backfill=allow_completing_source_backfill,
            allow_completing_fragment_merges=allow_fragment_completion,
            allow_source_backfill=source_backfill_enabled,
            allow_seed_source_backfill=allow_seed_source_backfill,
            allow_completing_seed_source_backfill=(allow_seed_completion),
            allow_fragment_merges=allow_fragment_merges,
            edge_order=teacher_edge_order,
            teacher_action_filter=teacher_action_filter,
            edge_feature_index=edge_feature_index,
            teacher_feature_gate=teacher_feature_gate,
            target_extension_feature_gate=target_extension_feature_gate,
            seed_source_feature_gate=seed_source_feature_gate,
            min_component_observations=min_component_observations,
            max_applied_edits=max_applied_edits,
            max_target_extension_edits=max_target_extension_edits,
            max_source_backfill_edits=max_source_backfill_edits,
            max_seed_source_backfill_edits=max_seed_source_backfill_edits,
            max_fragment_merge_edits=max_fragment_merge_edits,
            max_completing_rescue_edits=max_completing_rescue_edits,
        )
        scores = _score_prediction_against_reference(
            rescue.tracks, reference, config=policy_config
        )
        applied = int(sum(int(row["applied"]) for row in rescue.rows))
        candidates = int(len(rescue.rows))
        scores = {
            **scores,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_teacher_adjacent_candidates": candidates,
            "track2p_teacher_adjacent_applied": applied,
            "track2p_teacher_adjacent_allow_completing_rescue": int(
                allow_completing_rescue
            ),
            "track2p_teacher_adjacent_allow_teacher_complete_row_rescue": int(
                allow_teacher_complete_row_rescue
            ),
            "track2p_teacher_adjacent_allow_teacher_supported_completion": int(
                allow_teacher_supported_completion
            ),
            "track2p_teacher_adjacent_allow_teacher_supported_completing_rescue": int(
                allow_teacher_supported_completing_rescue
            ),
            "track2p_teacher_adjacent_allow_teacher_confirmed_completing_rescue": int(
                allow_teacher_confirmed_completing_rescue
            ),
            "track2p_teacher_adjacent_allow_teacher_completion_gate": int(
                allow_teacher_completion_gate_enabled
            ),
            "track2p_teacher_adjacent_allow_completing_source_backfill": int(
                allow_completing_source_backfill
            ),
            "track2p_teacher_adjacent_allow_completing_fragment_merge": int(
                allow_completing_fragment_merge
            ),
            "track2p_teacher_adjacent_allow_completing_fragment_merges": int(
                allow_fragment_completion
            ),
            "track2p_teacher_adjacent_allow_source_backfill": int(
                source_backfill_enabled
            ),
            "track2p_teacher_adjacent_allow_source_inserts": int(
                source_backfill_enabled
            ),
            "track2p_teacher_adjacent_allow_source_insertions": int(
                source_backfill_enabled
            ),
            "track2p_teacher_adjacent_allow_seed_source_backfill": int(
                allow_seed_source_backfill
            ),
            "track2p_teacher_adjacent_allow_seed_completing_rescue": int(
                allow_seed_completing_rescue
            ),
            "track2p_teacher_adjacent_allow_seed_completing_backfill": int(
                allow_seed_completing_backfill
            ),
            "track2p_teacher_adjacent_allow_completing_seed_source_backfill": int(
                allow_seed_completion
            ),
            "track2p_teacher_adjacent_allow_fragment_merges": int(
                allow_fragment_merges
            ),
            "track2p_teacher_adjacent_edge_order": str(teacher_edge_order),
            "track2p_teacher_adjacent_action_filter": str(teacher_action_filter),
            "track2p_teacher_adjacent_min_component_observations": int(
                min_component_observations
            ),
            "track2p_teacher_adjacent_max_applied_edits": (
                -1 if max_applied_edits is None else int(max_applied_edits)
            ),
            "track2p_teacher_adjacent_max_target_extension_edits": (
                -1
                if max_target_extension_edits is None
                else int(max_target_extension_edits)
            ),
            "track2p_teacher_adjacent_max_source_backfill_edits": (
                -1
                if max_source_backfill_edits is None
                else int(max_source_backfill_edits)
            ),
            "track2p_teacher_adjacent_max_seed_source_backfill_edits": (
                -1
                if max_seed_source_backfill_edits is None
                else int(max_seed_source_backfill_edits)
            ),
            "track2p_teacher_adjacent_max_fragment_merge_edits": (
                -1
                if max_fragment_merge_edits is None
                else int(max_fragment_merge_edits)
            ),
            "track2p_teacher_adjacent_max_completing_rescue_edits": (
                -1
                if max_completing_rescue_edits is None
                else int(max_completing_rescue_edits)
            ),
            "track2p_teacher_adjacent_repair_preset": str(teacher_repair_preset),
            "track2p_teacher_adjacent_feature_preset": str(teacher_feature_preset),
            "track2p_teacher_adjacent_target_extension_feature_preset": str(
                target_extension_feature_preset
            ),
            "track2p_teacher_adjacent_seed_source_feature_preset": str(
                seed_source_feature_preset
            ),
            "track2p_teacher_adjacent_action_specific_feature_gate_enabled": int(
                _teacher_feature_gate_enabled(target_extension_feature_gate)
                or _teacher_feature_gate_enabled(seed_source_feature_gate)
            ),
            "track2p_teacher_adjacent_feature_gate_enabled": int(
                _teacher_feature_gate_enabled(teacher_feature_gate)
            ),
            "track2p_teacher_adjacent_min_registered_iou": _score_optional_float(
                None
                if teacher_feature_gate is None
                else teacher_feature_gate.min_registered_iou
            ),
            "track2p_teacher_adjacent_max_registered_iou": _score_optional_float(
                None
                if teacher_feature_gate is None
                else teacher_feature_gate.max_registered_iou
            ),
            "track2p_teacher_adjacent_min_threshold_margin": _score_optional_float(
                None
                if teacher_feature_gate is None
                else teacher_feature_gate.min_threshold_margin
            ),
            "track2p_teacher_adjacent_min_row_margin": _score_optional_float(
                None
                if teacher_feature_gate is None
                else teacher_feature_gate.min_row_margin
            ),
            "track2p_teacher_adjacent_min_column_margin": _score_optional_float(
                None
                if teacher_feature_gate is None
                else teacher_feature_gate.min_column_margin
            ),
            "track2p_teacher_adjacent_max_centroid_distance": _score_optional_float(
                None
                if teacher_feature_gate is None
                else teacher_feature_gate.max_centroid_distance
            ),
            "track2p_teacher_adjacent_min_area_ratio": _score_optional_float(
                None
                if teacher_feature_gate is None
                else teacher_feature_gate.min_area_ratio
            ),
            "track2p_teacher_adjacent_min_cell_probability": _score_optional_float(
                None
                if teacher_feature_gate is None
                else teacher_feature_gate.min_cell_probability
            ),
            "track2p_teacher_adjacent_require_hungarian": int(
                teacher_feature_gate is not None
                and teacher_feature_gate.require_hungarian
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant="Track2p-policy component cleanup + teacher adjacent rescue",
                method=cast(Any, TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        rescue_rows.extend(
            {
                **row,
                "subject": subject_dir.name,
                "threshold_method": str(threshold_method),
                "iou_distance_threshold": f"{float(iou_distance_threshold):g}",
                "cell_probability_threshold": f"{float(policy_config.cell_probability_threshold):g}",
                "transform_type": str(policy_config.transform_type),
            }
            for row in rescue.rows
        )
    return ComponentAuditOutput(tuple(results), tuple(rescue_rows))


def _component_cleanup_prediction(
    sessions: Sequence[Track2pSession],
    reference_tracks: np.ndarray,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> np.ndarray:
    prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    policy_full = _normalize_int_track_matrix(prediction.tracks)
    policy_eval, reference_eval, evaluated_track_ids = _evaluated_prediction_rows(
        policy_full, reference_tracks, config=config
    )
    audit_rows = component_audit_rows(
        policy_eval,
        reference_eval,
        sessions=sessions,
        diagnostics=prediction.diagnostics,
        subject="",
        config=cleanup_config,
        track_ids=evaluated_track_ids,
        seed_session=config.seed_session,
    )
    return apply_weakest_bridge_splits(
        policy_full, _mark_applied_splits(audit_rows, apply_splits=True)
    )


def apply_teacher_adjacent_rescue_edges(
    predicted_track_matrix: Any,
    teacher_track_matrix: Any,
    *,
    seed_session: int = 0,
    allow_completing_rescue: bool = False,
    allow_teacher_complete_row_rescue: bool = False,
    allow_teacher_supported_completion: bool = False,
    allow_teacher_supported_completing_rescue: bool = False,
    allow_teacher_confirmed_completing_rescue: bool = False,
    allow_completing_source_backfill: bool = False,
    allow_completing_fragment_merge: bool = False,
    allow_completing_fragment_merges: bool = False,
    allow_source_backfill: bool = True,
    allow_source_inserts: bool | None = None,
    allow_source_insertions: bool | None = None,
    allow_seed_source_backfill: bool = False,
    allow_seed_completing_backfill: bool = False,
    allow_seed_completing_rescue: bool = False,
    allow_completing_seed_source_backfill: bool = False,
    allow_fragment_merges: bool = True,
    edge_order: TeacherEdgeOrder = "structural",
    teacher_action_filter: TeacherActionFilter = "all",
    edge_feature_index: Mapping[TrackEdge, ResidualFeature] | None = None,
    teacher_feature_gate: TeacherEdgeFeatureGate | None = None,
    feature_gate: TeacherEdgeFeatureGate | None = None,
    edge_feature_gate: TeacherEdgeFeatureGate | None = None,
    target_extension_feature_gate: TeacherEdgeFeatureGate | None = None,
    seed_source_feature_gate: TeacherEdgeFeatureGate | None = None,
    min_component_observations: int = 1,
    max_applied_edits: int | None = None,
    max_target_extension_edits: int | None = None,
    max_source_backfill_edits: int | None = None,
    max_seed_source_backfill_edits: int | None = None,
    max_fragment_merge_edits: int | None = None,
    max_completing_rescue_edits: int | None = None,
) -> TeacherAdjacentRescueReport:
    """Apply conflict-free adjacent Track2p-teacher edits.

    The operation is intentionally conservative: it can insert a missing target,
    insert a missing source into an already seed-anchored component, or merge two
    compatible seed-anchored fragments. It still rejects edits that would complete
    a row unless completion is explicitly enabled for the corresponding rescue
    path, or unless the completed row is itself present as a complete Track2p
    teacher row and ``allow_teacher_supported_completing_rescue`` is enabled.
    ``allow_teacher_confirmed_completing_rescue`` is a compatibility alias for
    the same exact teacher-row completion gate.

    ``allow_completing_fragment_merges`` is a narrower opt-in: it allows only
    complete-row fragment merges, while still rejecting single-edge target/source
    insertions that would complete a row.

    Teacher edges are structurally ordered by default so edges that merge or
    backfill already supported fragments are tried before plain forward extensions.
    This avoids lexicographic ROI ordering consuming an empty slot with a weaker
    teacher edge before a stronger bridge/backfill edge is tested. The dynamic
    structural order recomputes that priority after every attempted teacher edit,
    allowing a newly inserted source/target observation to promote a now-available
    fragment merge before weaker stale candidates consume slots. The
    ``confidence`` order keeps the same structural action classes, but breaks ties
    with label-free local registration evidence. ``dynamic-confidence`` combines
    both ideas: it recomputes the structural priority after each attempted edit and
    uses local registration evidence to break ties among currently eligible edits.
    This is useful for residual teacher-rescue sweeps because a wrong early teacher
    edge can claim the only open source/target slot and prevent a stronger edge
    from being tested later.
    ``dynamic-seed-confidence`` keeps the dynamic confidence machinery, but tries
    eligible seed-source backfills before other source-backfill edits so a small
    ``max_applied_edits`` cap is spent on the seed-anchoring hypothesis first.
    ``min_component_observations`` is a label-free support gate: teacher edits
    must touch at least one component with this many existing observations.

    ``max_applied_edits`` caps the number of accepted teacher edits per subject.
    This makes it possible to test the high-confidence first-edit regime without
    admitting a long tail of Track2p-teacher edges after the best rescue.
    Per-action caps can additionally reserve the small edit budget across the
    residual action families, e.g. one missing-seed source backfill plus two
    target extensions.  They are evaluated on the current dynamic action of the
    edge, so a target extension that would also complete a row can be limited by
    either its target-extension cap or the completing-rescue cap.
    """

    allow_completing_rescue = _bool_value(
        allow_completing_rescue, name="allow_completing_rescue"
    )
    allow_teacher_complete_row_rescue = _bool_value(
        allow_teacher_complete_row_rescue,
        name="allow_teacher_complete_row_rescue",
    )
    allow_teacher_supported_completion = _bool_value(
        allow_teacher_supported_completion,
        name="allow_teacher_supported_completion",
    )
    allow_teacher_supported_completing_rescue = _bool_value(
        allow_teacher_supported_completing_rescue,
        name="allow_teacher_supported_completing_rescue",
    )
    allow_teacher_confirmed_completing_rescue = _bool_value(
        allow_teacher_confirmed_completing_rescue,
        name="allow_teacher_confirmed_completing_rescue",
    )
    allow_completing_source_backfill = _bool_value(
        allow_completing_source_backfill,
        name="allow_completing_source_backfill",
    )
    allow_completing_fragment_merge = _bool_value(
        allow_completing_fragment_merge,
        name="allow_completing_fragment_merge",
    )
    allow_completing_fragment_merges = _bool_value(
        allow_completing_fragment_merges,
        name="allow_completing_fragment_merges",
    )
    allow_source_backfill = _bool_value(
        allow_source_backfill, name="allow_source_backfill"
    )
    allow_source_inserts = _optional_bool_value(
        allow_source_inserts, name="allow_source_inserts"
    )
    allow_source_insertions = _optional_bool_value(
        allow_source_insertions, name="allow_source_insertions"
    )
    allow_seed_source_backfill = _bool_value(
        allow_seed_source_backfill, name="allow_seed_source_backfill"
    )
    allow_seed_completing_backfill = _bool_value(
        allow_seed_completing_backfill,
        name="allow_seed_completing_backfill",
    )
    allow_seed_completing_rescue = _bool_value(
        allow_seed_completing_rescue, name="allow_seed_completing_rescue"
    )
    allow_completing_seed_source_backfill = _bool_value(
        allow_completing_seed_source_backfill,
        name="allow_completing_seed_source_backfill",
    )
    allow_fragment_merges = _bool_value(
        allow_fragment_merges, name="allow_fragment_merges"
    )
    output = _normalize_int_track_matrix(predicted_track_matrix)
    teacher = _normalize_int_track_matrix(teacher_track_matrix)
    allow_partial_teacher_completion = bool(
        allow_teacher_supported_completion or allow_teacher_supported_completing_rescue
    )
    allow_exact_teacher_completion = bool(
        allow_teacher_complete_row_rescue or allow_teacher_confirmed_completing_rescue
    )
    teacher_complete_tracks = (
        _teacher_row_set(teacher)
        if allow_partial_teacher_completion
        else _complete_row_set(teacher)
    )
    allow_teacher_supported_completion_enabled = bool(
        allow_partial_teacher_completion or allow_exact_teacher_completion
    )
    allow_seed_completion = bool(
        allow_seed_completing_backfill
        or allow_seed_completing_rescue
        or allow_completing_seed_source_backfill
    )
    allow_fragment_completion = bool(
        allow_completing_fragment_merge or allow_completing_fragment_merges
    )
    action_edit_caps = _normalized_action_edit_caps(
        max_target_extension_edits=max_target_extension_edits,
        max_source_backfill_edits=max_source_backfill_edits,
        max_seed_source_backfill_edits=max_seed_source_backfill_edits,
        max_fragment_merge_edits=max_fragment_merge_edits,
        max_completing_rescue_edits=max_completing_rescue_edits,
    )
    action_edit_counts = _initial_action_edit_counts(action_edit_caps)
    min_component_observations = _positive_int_value(
        min_component_observations, name="min_component_observations"
    )
    max_applied_edits = _normalized_max_applied_edits(max_applied_edits)
    teacher_feature_gate = _resolve_teacher_feature_gate(
        teacher_feature_gate, feature_gate, edge_feature_gate
    )
    target_extension_feature_gate = _resolve_teacher_feature_gate(
        target_extension_feature_gate
    )
    seed_source_feature_gate = _resolve_teacher_feature_gate(seed_source_feature_gate)
    source_backfill_enabled = _resolve_source_backfill_alias(
        allow_source_backfill, allow_source_inserts, allow_source_insertions
    )
    if edge_order in {
        "dynamic-structural",
        "dynamic-confidence",
        "dynamic-cell-confidence",
        "dynamic-seed-confidence",
        "dynamic-seed-cell-confidence",
    }:
        return _apply_teacher_adjacent_rescue_edges_dynamic(
            output,
            teacher,
            seed_session=seed_session,
            teacher_action_filter=teacher_action_filter,
            allow_completing_rescue=allow_completing_rescue,
            allow_teacher_supported_completing_rescue=(
                allow_teacher_supported_completion_enabled
            ),
            teacher_complete_tracks=teacher_complete_tracks,
            allow_completing_source_backfill=allow_completing_source_backfill,
            allow_completing_fragment_merges=allow_fragment_completion,
            allow_source_backfill=source_backfill_enabled,
            allow_seed_source_backfill=allow_seed_source_backfill,
            allow_completing_seed_source_backfill=(allow_seed_completion),
            allow_fragment_merges=allow_fragment_merges,
            min_component_observations=min_component_observations,
            edge_feature_index=edge_feature_index or {},
            use_confidence_order=edge_order
            in {
                "dynamic-confidence",
                "dynamic-cell-confidence",
                "dynamic-seed-confidence",
                "dynamic-seed-cell-confidence",
            },
            prefer_cell_confidence=edge_order
            in {"dynamic-cell-confidence", "dynamic-seed-cell-confidence"},
            prioritize_seed_source_backfill=edge_order
            in {"dynamic-seed-confidence", "dynamic-seed-cell-confidence"},
            teacher_feature_gate=teacher_feature_gate,
            target_extension_feature_gate=target_extension_feature_gate,
            seed_source_feature_gate=seed_source_feature_gate,
            max_applied_edits=max_applied_edits,
            action_edit_caps=action_edit_caps,
        )
    edge_occurrences = _ordered_teacher_edge_occurrences(
        output,
        teacher,
        edge_order=edge_order,
        seed_session=seed_session,
        allow_completing_rescue=allow_completing_rescue,
        allow_teacher_supported_completing_rescue=(
            allow_teacher_supported_completion_enabled
        ),
        teacher_complete_tracks=teacher_complete_tracks,
        allow_completing_source_backfill=allow_completing_source_backfill,
        allow_completing_fragment_merges=allow_fragment_completion,
        allow_source_backfill=source_backfill_enabled,
        allow_seed_source_backfill=allow_seed_source_backfill,
        allow_completing_seed_source_backfill=allow_seed_completion,
        allow_fragment_merges=allow_fragment_merges,
        edge_feature_index=edge_feature_index or {},
        min_component_observations=min_component_observations,
    )
    rows: list[dict[str, int | str]] = []
    applied_count = 0
    for edge, occurrence_index in edge_occurrences:
        if track_edge_counter(output).get(edge, 0) > occurrence_index:
            continue
        if _max_applied_edits_reached(applied_count, max_applied_edits):
            rows.append(
                {
                    **_teacher_edge_limit_row(edge),
                    "occurrence_index": int(occurrence_index),
                }
            )
            continue
        action_reason = _teacher_edge_action_filter_reason(
            output,
            edge,
            seed_session=seed_session,
            action_filter=teacher_action_filter,
        )
        if action_reason != "accepted":
            rows.append(
                {
                    **_teacher_edge_rejection_row(edge, action_reason),
                    "occurrence_index": int(occurrence_index),
                }
            )
            continue
        selected_feature_gate = _teacher_edge_feature_gate_for_action(
            output,
            edge,
            seed_session=seed_session,
            default_gate=teacher_feature_gate,
            target_extension_gate=target_extension_feature_gate,
            seed_source_gate=seed_source_feature_gate,
        )
        gate_reason = _teacher_edge_feature_gate_reason(
            (edge_feature_index or {}).get(edge), selected_feature_gate
        )
        if gate_reason != "accepted":
            rows.append(
                {
                    **_teacher_edge_rejection_row(edge, gate_reason),
                    "occurrence_index": int(occurrence_index),
                }
            )
            continue
        action_cap_keys = _teacher_action_edit_cap_keys(
            output, edge, seed_session=seed_session
        )
        cap_reason = _teacher_action_edit_cap_reason(
            action_edit_counts, action_edit_caps, action_cap_keys
        )
        if cap_reason is not None:
            rows.append(
                {
                    **_teacher_edge_limit_row(edge, reason=cap_reason),
                    "occurrence_index": int(occurrence_index),
                }
            )
            continue
        output, row = _try_apply_teacher_edge(
            output,
            edge,
            seed_session=seed_session,
            allow_completing_rescue=allow_completing_rescue,
            allow_teacher_supported_completing_rescue=(
                allow_teacher_supported_completion_enabled
            ),
            teacher_complete_tracks=teacher_complete_tracks,
            allow_completing_source_backfill=allow_completing_source_backfill,
            allow_completing_fragment_merges=allow_fragment_completion,
            allow_source_backfill=source_backfill_enabled,
            allow_seed_source_backfill=allow_seed_source_backfill,
            allow_completing_seed_source_backfill=(allow_seed_completion),
            allow_fragment_merges=allow_fragment_merges,
            min_component_observations=min_component_observations,
        )
        if int(row.get("applied", 0)):
            applied_count += 1
            _record_teacher_action_edit(action_edit_counts, action_cap_keys)
        rows.append(
            {
                **row,
                "occurrence_index": int(occurrence_index),
            }
        )
    return TeacherAdjacentRescueReport(output, tuple(rows))


def _apply_teacher_adjacent_rescue_edges_dynamic(
    predicted: np.ndarray,
    teacher: np.ndarray,
    *,
    seed_session: int,
    teacher_action_filter: TeacherActionFilter,
    allow_completing_rescue: bool,
    allow_teacher_supported_completing_rescue: bool,
    teacher_complete_tracks: frozenset[tuple[int, ...]],
    allow_completing_source_backfill: bool,
    allow_completing_fragment_merges: bool,
    allow_source_backfill: bool,
    allow_seed_source_backfill: bool,
    allow_completing_seed_source_backfill: bool,
    allow_fragment_merges: bool,
    min_component_observations: int,
    edge_feature_index: Mapping[TrackEdge, ResidualFeature],
    use_confidence_order: bool,
    prefer_cell_confidence: bool,
    prioritize_seed_source_backfill: bool,
    teacher_feature_gate: TeacherEdgeFeatureGate | None,
    target_extension_feature_gate: TeacherEdgeFeatureGate | None,
    seed_source_feature_gate: TeacherEdgeFeatureGate | None,
    max_applied_edits: int | None,
    action_edit_caps: Mapping[str, int],
) -> TeacherAdjacentRescueReport:
    """Apply teacher edits while recomputing structural priorities."""

    output = np.asarray(predicted, dtype=int).copy()
    occurrences = tuple(
        (edge, occurrence_index)
        for edge, count in sorted(track_edge_counter(teacher).items())
        for occurrence_index in range(int(count))
    )
    attempted: set[tuple[TrackEdge, int]] = set()
    rows: list[dict[str, int | str]] = []
    applied_count = 0
    action_edit_counts = _initial_action_edit_counts(action_edit_caps)

    while True:
        output_counts = track_edge_counter(output)
        pending: list[tuple[TrackEdge, int]] = []
        deferred_action_rows: list[dict[str, int | str]] = []
        for edge, occurrence_index in occurrences:
            occurrence = (edge, occurrence_index)
            if occurrence in attempted:
                continue
            if output_counts.get(edge, 0) > occurrence_index:
                attempted.add(occurrence)
                continue
            action_reason = _teacher_edge_action_filter_reason(
                output,
                edge,
                seed_session=seed_session,
                action_filter=teacher_action_filter,
            )
            if action_reason != "accepted":
                # Dynamic rescue changes the predicted matrix after each accepted
                # edit.  An edge that is not in the requested action class now can
                # become a valid target-extension/source-backfill/fragment-merge
                # after an earlier teacher edit inserts one of its endpoints.  Do
                # not mark action-filter misses as permanently attempted until the
                # loop reaches a fixed point with no currently eligible edge.
                deferred_action_rows.append(
                    {
                        **_teacher_edge_rejection_row(edge, action_reason),
                        "occurrence_index": int(occurrence_index),
                    }
                )
                continue
            selected_feature_gate = _teacher_edge_feature_gate_for_action(
                output,
                edge,
                seed_session=seed_session,
                default_gate=teacher_feature_gate,
                target_extension_gate=target_extension_feature_gate,
                seed_source_gate=seed_source_feature_gate,
            )
            gate_reason = _teacher_edge_feature_gate_reason(
                edge_feature_index.get(edge), selected_feature_gate
            )
            if gate_reason != "accepted":
                attempted.add(occurrence)
                rows.append(
                    {
                        **_teacher_edge_rejection_row(edge, gate_reason),
                        "occurrence_index": int(occurrence_index),
                    }
                )
                continue
            pending.append(occurrence)
        if not pending:
            rows.extend(deferred_action_rows)
            break

        edge, occurrence_index = min(
            pending,
            key=lambda item: _teacher_edge_dynamic_order_key(
                output,
                item[0],
                seed_session=seed_session,
                allow_completing_rescue=allow_completing_rescue,
                allow_teacher_supported_completing_rescue=(
                    allow_teacher_supported_completing_rescue
                ),
                teacher_complete_tracks=teacher_complete_tracks,
                allow_completing_source_backfill=allow_completing_source_backfill,
                allow_completing_fragment_merges=allow_completing_fragment_merges,
                allow_source_backfill=allow_source_backfill,
                allow_seed_source_backfill=allow_seed_source_backfill,
                allow_completing_seed_source_backfill=(
                    allow_completing_seed_source_backfill
                ),
                allow_fragment_merges=allow_fragment_merges,
                edge_feature_index=edge_feature_index,
                min_component_observations=min_component_observations,
                use_confidence_order=use_confidence_order,
                prefer_cell_confidence=prefer_cell_confidence,
                prioritize_seed_source_backfill=prioritize_seed_source_backfill,
                occurrence_index=item[1],
            ),
        )
        attempted.add((edge, occurrence_index))
        if _max_applied_edits_reached(applied_count, max_applied_edits):
            rows.append(
                {
                    **_teacher_edge_limit_row(edge),
                    "occurrence_index": int(occurrence_index),
                }
            )
            continue
        action_cap_keys = _teacher_action_edit_cap_keys(
            output, edge, seed_session=seed_session
        )
        cap_reason = _teacher_action_edit_cap_reason(
            action_edit_counts, action_edit_caps, action_cap_keys
        )
        if cap_reason is not None:
            rows.append(
                {
                    **_teacher_edge_limit_row(edge, reason=cap_reason),
                    "occurrence_index": int(occurrence_index),
                }
            )
            continue
        output, row = _try_apply_teacher_edge(
            output,
            edge,
            seed_session=seed_session,
            allow_completing_rescue=allow_completing_rescue,
            allow_teacher_supported_completing_rescue=(
                allow_teacher_supported_completing_rescue
            ),
            teacher_complete_tracks=teacher_complete_tracks,
            allow_completing_source_backfill=allow_completing_source_backfill,
            allow_completing_fragment_merges=allow_completing_fragment_merges,
            allow_source_backfill=allow_source_backfill,
            allow_seed_source_backfill=allow_seed_source_backfill,
            allow_completing_seed_source_backfill=(
                allow_completing_seed_source_backfill
            ),
            allow_fragment_merges=allow_fragment_merges,
            min_component_observations=min_component_observations,
        )
        if int(row.get("applied", 0)):
            applied_count += 1
            _record_teacher_action_edit(action_edit_counts, action_cap_keys)
        rows.append({**row, "occurrence_index": int(occurrence_index)})

    return TeacherAdjacentRescueReport(output, tuple(rows))


def _teacher_action_filter_rejection_may_change_after_edit(reason: str) -> bool:
    """Return whether a dynamic teacher edit can make this rejection stale."""

    # Action classes are state-dependent: after one teacher edge inserts a source,
    # the next edge from that source can become a target extension or completion.
    return str(reason).startswith("action_filter_")


def _normalized_max_applied_edits(max_applied_edits: int | None) -> int | None:
    if max_applied_edits is None:
        return None
    return _nonnegative_int_value(max_applied_edits, name="max_applied_edits")


def _max_applied_edits_reached(
    applied_count: int, max_applied_edits: int | None
) -> bool:
    return max_applied_edits is not None and int(applied_count) >= int(
        max_applied_edits
    )


def _normalized_action_edit_caps(
    *,
    max_target_extension_edits: int | None,
    max_source_backfill_edits: int | None,
    max_seed_source_backfill_edits: int | None,
    max_fragment_merge_edits: int | None,
    max_completing_rescue_edits: int | None,
) -> dict[str, int]:
    caps: dict[str, int] = {}
    for key, value in (
        ("target_extension", max_target_extension_edits),
        ("source_backfill", max_source_backfill_edits),
        ("seed_source_backfill", max_seed_source_backfill_edits),
        ("fragment_merge", max_fragment_merge_edits),
        ("completing_rescue", max_completing_rescue_edits),
    ):
        if value is not None:
            caps[key] = _nonnegative_int_value(value, name=f"max_{key}_edits")
    return caps


def _initial_action_edit_counts(action_edit_caps: Mapping[str, int]) -> dict[str, int]:
    return {key: 0 for key in action_edit_caps}


def _teacher_action_edit_cap_keys(
    predicted: np.ndarray, edge: TrackEdge, *, seed_session: int
) -> tuple[str, ...]:
    action = _teacher_edge_action(predicted, edge, seed_session=seed_session)
    keys: list[str] = []
    if action in {
        "target-extension",
        "source-backfill",
        "seed-source-backfill",
        "fragment-merge",
    }:
        keys.append(action.replace("-", "_"))
    if _teacher_edge_would_complete_row(predicted, edge):
        keys.append("completing_rescue")
    return tuple(dict.fromkeys(keys))


def _teacher_action_edit_cap_reason(
    action_edit_counts: Mapping[str, int],
    action_edit_caps: Mapping[str, int],
    action_cap_keys: Sequence[str],
) -> str | None:
    for key in action_cap_keys:
        if key in action_edit_caps and int(action_edit_counts.get(key, 0)) >= int(
            action_edit_caps[key]
        ):
            return f"max_{key}_edits_reached"
    return None


def _record_teacher_action_edit(
    action_edit_counts: dict[str, int], action_cap_keys: Sequence[str]
) -> None:
    for key in action_cap_keys:
        if key in action_edit_counts:
            action_edit_counts[key] = int(action_edit_counts[key]) + 1


def _teacher_edge_limit_row(
    edge: TrackEdge, *, reason: str = "max_applied_edits_reached"
) -> dict[str, int | str]:
    session_a, session_b, roi_a, roi_b = edge
    return {
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "applied": 0,
        "reason": reason,
        "source_row": -1,
        "target_row": -1,
        "teacher_complete_row_supported": 0,
    }


def _ordered_teacher_edge_occurrences(
    predicted: np.ndarray,
    teacher: np.ndarray,
    *,
    edge_order: TeacherEdgeOrder,
    seed_session: int,
    allow_completing_rescue: bool,
    allow_teacher_supported_completing_rescue: bool,
    teacher_complete_tracks: frozenset[tuple[int, ...]],
    allow_completing_source_backfill: bool,
    allow_completing_fragment_merges: bool,
    allow_source_backfill: bool,
    allow_seed_source_backfill: bool,
    allow_completing_seed_source_backfill: bool,
    allow_fragment_merges: bool,
    edge_feature_index: Mapping[TrackEdge, ResidualFeature],
    min_component_observations: int,
) -> tuple[tuple[TrackEdge, int], ...]:
    occurrences = tuple(
        (edge, occurrence_index)
        for edge, count in sorted(track_edge_counter(teacher).items())
        for occurrence_index in range(int(count))
    )
    if edge_order == "lexicographic":
        return occurrences
    if edge_order in {
        "dynamic-structural",
        "dynamic-confidence",
        "dynamic-cell-confidence",
        "dynamic-seed-confidence",
        "dynamic-seed-cell-confidence",
    }:
        return occurrences
    if edge_order in {"confidence", "cell-confidence"}:
        return tuple(
            sorted(
                occurrences,
                key=lambda item: (
                    _teacher_edge_confidence_order_key(
                        predicted,
                        item[0],
                        seed_session=seed_session,
                        allow_completing_rescue=allow_completing_rescue,
                        allow_teacher_supported_completing_rescue=(
                            allow_teacher_supported_completing_rescue
                        ),
                        teacher_complete_tracks=teacher_complete_tracks,
                        allow_completing_source_backfill=(
                            allow_completing_source_backfill
                        ),
                        allow_completing_fragment_merges=(
                            allow_completing_fragment_merges
                        ),
                        allow_source_backfill=allow_source_backfill,
                        allow_seed_source_backfill=allow_seed_source_backfill,
                        allow_completing_seed_source_backfill=(
                            allow_completing_seed_source_backfill
                        ),
                        allow_fragment_merges=allow_fragment_merges,
                        edge_feature_index=edge_feature_index,
                        min_component_observations=min_component_observations,
                        prefer_cell_confidence=edge_order == "cell-confidence",
                    ),
                    item[1],
                ),
            )
        )
    if edge_order != "structural":
        raise ValueError(f"Unsupported teacher edge order: {edge_order!r}")
    return tuple(
        sorted(
            occurrences,
            key=lambda item: (
                _teacher_edge_structural_order_key(
                    predicted,
                    item[0],
                    seed_session=seed_session,
                    allow_completing_rescue=allow_completing_rescue,
                    allow_teacher_supported_completing_rescue=(
                        allow_teacher_supported_completing_rescue
                    ),
                    teacher_complete_tracks=teacher_complete_tracks,
                    allow_completing_source_backfill=(allow_completing_source_backfill),
                    allow_completing_fragment_merges=(allow_completing_fragment_merges),
                    allow_source_backfill=allow_source_backfill,
                    allow_seed_source_backfill=allow_seed_source_backfill,
                    allow_completing_seed_source_backfill=(
                        allow_completing_seed_source_backfill
                    ),
                    allow_fragment_merges=allow_fragment_merges,
                    min_component_observations=min_component_observations,
                ),
                item[1],
            ),
        )
    )


def _teacher_edge_structural_order_key(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    seed_session: int,
    allow_completing_rescue: bool,
    allow_teacher_supported_completing_rescue: bool,
    teacher_complete_tracks: frozenset[tuple[int, ...]],
    allow_completing_source_backfill: bool,
    allow_completing_fragment_merges: bool,
    allow_source_backfill: bool,
    allow_seed_source_backfill: bool,
    allow_completing_seed_source_backfill: bool,
    allow_fragment_merges: bool,
    min_component_observations: int,
) -> tuple[int, int, int, int, int, int]:
    """Rank Track2p teacher edits by expected structural value.

    Lower tuples are tried first. The ranking is label-free and uses only the
    current predicted matrix: opt-in seed-session source backfills outrank
    compatible fragment merges, which outrank ordinary source backfills and then
    plain target extensions. The seed-backfill priority only applies when
    ``allow_seed_source_backfill`` is enabled and directly targets the residual
    missing-seed bucket. Within an action class, edits involving more
    already-supported observations are tried first.
    """

    session_a, session_b, roi_a, roi_b = edge
    source_rows = tuple(np.flatnonzero(predicted[:, session_a] == roi_a))
    target_rows = tuple(np.flatnonzero(predicted[:, session_b] == roi_b))
    source_row = int(source_rows[0]) if len(source_rows) == 1 else -1
    target_row = int(target_rows[0]) if len(target_rows) == 1 else -1
    source_support = (
        _component_observation_count(predicted[source_row]) if source_row >= 0 else 0
    )
    target_support = (
        _component_observation_count(predicted[target_row]) if target_row >= 0 else 0
    )

    action_rank = 9
    evidence = 0
    if source_row >= 0 and target_row >= 0:
        if source_row != target_row and allow_fragment_merges:
            merged = _merge_rows_if_compatible(
                predicted[source_row], predicted[target_row]
            )
            if (
                merged is not None
                and _row_is_seed_anchored(merged, seed_session)
                and max(source_support, target_support) >= min_component_observations
                and not _would_complete_track(
                    merged,
                    bool(allow_completing_rescue or allow_completing_fragment_merges),
                    allow_teacher_supported_completing_rescue=(
                        allow_teacher_supported_completing_rescue
                    ),
                    teacher_complete_tracks=teacher_complete_tracks,
                )
            ):
                action_rank = 1
                evidence = int(np.count_nonzero(merged >= 0))
    elif source_row < 0 and target_row >= 0:
        target_seed_anchored = _row_is_seed_anchored(
            predicted[target_row], seed_session
        )
        seed_source_backfill = bool(
            allow_seed_source_backfill and session_a == seed_session
        )
        source_backfill_allowed = _source_backfill_allowed_for_edge(
            allow_source_backfill,
            allow_seed_source_backfill=allow_seed_source_backfill,
            session_a=int(session_a),
            seed_session=int(seed_session),
        )
        if (
            (source_backfill_allowed and target_seed_anchored)
            or seed_source_backfill
            or (source_backfill_allowed and allow_teacher_supported_completing_rescue)
        ):
            candidate = predicted[target_row].copy()
            candidate[session_a] = roi_a
            has_target_support = target_support >= min_component_observations
            allow_completion = bool(
                allow_completing_rescue
                or allow_completing_source_backfill
                or (seed_source_backfill and allow_completing_seed_source_backfill)
            )
            if has_target_support and not _would_complete_track(
                candidate,
                allow_completion,
                allow_teacher_supported_completing_rescue=(
                    allow_teacher_supported_completing_rescue
                ),
                teacher_complete_tracks=teacher_complete_tracks,
            ):
                action_rank = 0 if seed_source_backfill else 2
                evidence = int(np.count_nonzero(candidate >= 0))
    elif source_row >= 0 and target_row < 0:
        if _row_is_seed_anchored(predicted[source_row], seed_session):
            candidate = predicted[source_row].copy()
            candidate[session_b] = roi_b
            has_source_support = source_support >= min_component_observations
            if has_source_support and not _would_complete_track(
                candidate,
                allow_completing_rescue,
                allow_teacher_supported_completing_rescue=(
                    allow_teacher_supported_completing_rescue
                ),
                teacher_complete_tracks=teacher_complete_tracks,
            ):
                action_rank = 3
                evidence = int(np.count_nonzero(candidate >= 0))

    return (
        action_rank,
        -evidence,
        int(session_a),
        int(session_b),
        int(roi_a),
        int(roi_b),
    )


def _teacher_edge_dynamic_order_key(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    seed_session: int,
    allow_completing_rescue: bool,
    allow_teacher_supported_completing_rescue: bool,
    teacher_complete_tracks: frozenset[tuple[int, ...]],
    allow_completing_source_backfill: bool,
    allow_completing_fragment_merges: bool,
    allow_source_backfill: bool,
    allow_seed_source_backfill: bool,
    allow_completing_seed_source_backfill: bool,
    allow_fragment_merges: bool,
    edge_feature_index: Mapping[TrackEdge, ResidualFeature],
    min_component_observations: int,
    use_confidence_order: bool,
    prefer_cell_confidence: bool,
    prioritize_seed_source_backfill: bool,
    occurrence_index: int,
) -> tuple[Any, ...]:
    """Return the dynamic rescue priority for one pending teacher edge."""

    if use_confidence_order:
        return (
            _teacher_edge_confidence_order_key(
                predicted,
                edge,
                seed_session=seed_session,
                allow_completing_rescue=allow_completing_rescue,
                allow_teacher_supported_completing_rescue=(
                    allow_teacher_supported_completing_rescue
                ),
                teacher_complete_tracks=teacher_complete_tracks,
                allow_completing_source_backfill=allow_completing_source_backfill,
                allow_completing_fragment_merges=allow_completing_fragment_merges,
                allow_source_backfill=allow_source_backfill,
                allow_seed_source_backfill=allow_seed_source_backfill,
                allow_completing_seed_source_backfill=(
                    allow_completing_seed_source_backfill
                ),
                allow_fragment_merges=allow_fragment_merges,
                edge_feature_index=edge_feature_index,
                min_component_observations=min_component_observations,
                prefer_cell_confidence=prefer_cell_confidence,
                prioritize_seed_source_backfill=prioritize_seed_source_backfill,
            ),
            int(occurrence_index),
        )
    return (
        _teacher_edge_structural_order_key(
            predicted,
            edge,
            seed_session=seed_session,
            allow_completing_rescue=allow_completing_rescue,
            allow_teacher_supported_completing_rescue=(
                allow_teacher_supported_completing_rescue
            ),
            teacher_complete_tracks=teacher_complete_tracks,
            allow_completing_source_backfill=allow_completing_source_backfill,
            allow_completing_fragment_merges=allow_completing_fragment_merges,
            allow_source_backfill=allow_source_backfill,
            allow_seed_source_backfill=allow_seed_source_backfill,
            allow_completing_seed_source_backfill=allow_completing_seed_source_backfill,
            allow_fragment_merges=allow_fragment_merges,
            min_component_observations=min_component_observations,
        ),
        int(occurrence_index),
    )


def _teacher_edge_confidence_order_key(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    seed_session: int,
    allow_completing_rescue: bool,
    allow_teacher_supported_completing_rescue: bool,
    teacher_complete_tracks: frozenset[tuple[int, ...]],
    allow_completing_source_backfill: bool,
    allow_completing_fragment_merges: bool,
    allow_source_backfill: bool,
    allow_seed_source_backfill: bool,
    allow_completing_seed_source_backfill: bool,
    allow_fragment_merges: bool,
    edge_feature_index: Mapping[TrackEdge, ResidualFeature],
    min_component_observations: int,
    prefer_cell_confidence: bool = False,
    prioritize_seed_source_backfill: bool = False,
) -> tuple[Any, ...]:
    """Return a label-free confidence-aware order key for teacher edges."""

    structural = _teacher_edge_structural_order_key(
        predicted,
        edge,
        seed_session=seed_session,
        allow_completing_rescue=allow_completing_rescue,
        allow_teacher_supported_completing_rescue=(
            allow_teacher_supported_completing_rescue
        ),
        teacher_complete_tracks=teacher_complete_tracks,
        allow_completing_source_backfill=allow_completing_source_backfill,
        allow_completing_fragment_merges=allow_completing_fragment_merges,
        allow_source_backfill=allow_source_backfill,
        allow_seed_source_backfill=allow_seed_source_backfill,
        allow_completing_seed_source_backfill=allow_completing_seed_source_backfill,
        allow_fragment_merges=allow_fragment_merges,
        min_component_observations=min_component_observations,
    )
    feature_key = _teacher_edge_feature_order_key(
        edge_feature_index.get(edge),
        prefer_cell_confidence=prefer_cell_confidence,
    )
    if prioritize_seed_source_backfill:
        seed_source_key = _teacher_edge_seed_source_backfill_order_key(
            predicted,
            edge,
            seed_session=seed_session,
            allow_source_backfill=allow_source_backfill,
            allow_seed_source_backfill=allow_seed_source_backfill,
            min_component_observations=min_component_observations,
        )
        return (
            *seed_source_key,
            structural[0],
            structural[1],
            *feature_key,
            structural[2],
            structural[3],
            structural[4],
            structural[5],
        )
    return (
        structural[0],
        structural[1],
        *feature_key,
        structural[2],
        structural[3],
        structural[4],
        structural[5],
    )


def _teacher_edge_seed_source_backfill_order_key(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    seed_session: int,
    allow_source_backfill: bool,
    allow_seed_source_backfill: bool,
    min_component_observations: int,
) -> tuple[int, int, int]:
    """Prefer eligible edits that backfill the missing seed-session source ROI."""

    session_a, session_b, roi_a, roi_b = edge
    if not _source_backfill_allowed_for_edge(
        allow_source_backfill,
        allow_seed_source_backfill=allow_seed_source_backfill,
        session_a=int(session_a),
        seed_session=int(seed_session),
    ):
        return (1, 0, 0)
    if int(session_a) != int(seed_session):
        return (1, 0, 0)
    source_rows = tuple(np.flatnonzero(predicted[:, session_a] == roi_a))
    target_rows = tuple(np.flatnonzero(predicted[:, session_b] == roi_b))
    if len(source_rows) != 0 or len(target_rows) != 1:
        return (1, 0, 0)
    target_row = int(target_rows[0])
    support = _component_observation_count(predicted[target_row])
    if support < min_component_observations:
        return (1, 0, 0)
    return (0, -int(support), int(session_b))


def _teacher_edge_feature_order_key(
    feature: ResidualFeature | None,
    *,
    prefer_cell_confidence: bool = False,
) -> tuple[int, float, float, float, float, float, float, float]:
    if prefer_cell_confidence:
        return _teacher_edge_cell_confidence_order_key(feature)
    if feature is None:
        return (1, 0.0, 0.0, 0.0, 0.0, float("inf"), 0.0, 0.0)
    registered_iou = _finite_feature(feature.registered_iou, 0.0)
    threshold_margin = _finite_feature(feature.threshold_margin, 0.0)
    row_margin = _finite_feature(feature.row_margin, 0.0)
    column_margin = _finite_feature(feature.column_margin, 0.0)
    min_cell_probability = min(
        _finite_feature(feature.cell_probability_a, 0.0),
        _finite_feature(feature.cell_probability_b, 0.0),
    )
    centroid_distance = _finite_feature(feature.centroid_distance, float("inf"))
    area_ratio = _finite_feature(feature.area_ratio, 0.0)
    return (
        0,
        -float(feature.assigned_by_hungarian),
        -registered_iou,
        -threshold_margin,
        -min(row_margin, column_margin),
        centroid_distance,
        -area_ratio,
        -min_cell_probability,
    )


def _teacher_edge_cell_confidence_order_key(
    feature: ResidualFeature | None,
) -> tuple[int, float, float, float, float, float, float, float]:
    """Order teacher edges by cell/shape confidence before raw registered IoU."""

    if feature is None:
        return (1, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0, 0.0)
    registered_iou = _finite_feature(feature.registered_iou, 0.0)
    threshold_margin = _finite_feature(feature.threshold_margin, 0.0)
    row_margin = _finite_feature(feature.row_margin, 0.0)
    column_margin = _finite_feature(feature.column_margin, 0.0)
    min_cell_probability = min(
        _finite_feature(feature.cell_probability_a, 0.0),
        _finite_feature(feature.cell_probability_b, 0.0),
    )
    centroid_distance = _finite_feature(feature.centroid_distance, float("inf"))
    area_ratio = _finite_feature(feature.area_ratio, 0.0)
    return (
        0,
        -float(feature.assigned_by_hungarian),
        -min_cell_probability,
        -area_ratio,
        centroid_distance,
        -threshold_margin,
        -min(row_margin, column_margin),
        -registered_iou,
    )


def _finite_feature(value: float, fallback: float) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else fallback


def _resolve_teacher_feature_gate(
    *gates: TeacherEdgeFeatureGate | None,
) -> TeacherEdgeFeatureGate | None:
    for gate in gates:
        if _teacher_feature_gate_enabled(gate):
            return gate
    return None


def _teacher_edge_feature_gate_for_action(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    seed_session: int,
    default_gate: TeacherEdgeFeatureGate | None,
    target_extension_gate: TeacherEdgeFeatureGate | None,
    seed_source_gate: TeacherEdgeFeatureGate | None,
) -> TeacherEdgeFeatureGate | None:
    """Return an action-specific feature gate when one is configured."""

    action = _teacher_edge_action(predicted, edge, seed_session=seed_session)
    if action == "target-extension" and _teacher_feature_gate_enabled(
        target_extension_gate
    ):
        return target_extension_gate
    if action == "seed-source-backfill" and _teacher_feature_gate_enabled(
        seed_source_gate
    ):
        return seed_source_gate
    return default_gate


def _teacher_feature_gate_enabled(gate: TeacherEdgeFeatureGate | None) -> bool:
    return bool(gate is not None and gate.enabled)


def _teacher_edge_feature_gate_reason(
    feature: ResidualFeature | None,
    gate: TeacherEdgeFeatureGate | None,
) -> str:
    if not _teacher_feature_gate_enabled(gate):
        return "accepted"
    if feature is None or gate is None:
        return "feature_gate_missing"
    if gate.require_hungarian and int(feature.assigned_by_hungarian) <= 0:
        return "feature_gate_hungarian"
    for value, threshold, name in (
        (feature.registered_iou, gate.min_registered_iou, "registered_iou"),
        (feature.threshold_margin, gate.min_threshold_margin, "threshold_margin"),
        (feature.row_margin, gate.min_row_margin, "row_margin"),
        (feature.column_margin, gate.min_column_margin, "column_margin"),
        (feature.area_ratio, gate.min_area_ratio, "area_ratio"),
    ):
        reason = _feature_min_reason(value, threshold, name)
        if reason is not None:
            return reason
    max_registered_iou = _feature_max_reason(
        feature.registered_iou, gate.max_registered_iou, "max_registered_iou"
    )
    if max_registered_iou is not None:
        return max_registered_iou
    min_cell = _feature_min_pair_reason(
        feature.cell_probability_a,
        feature.cell_probability_b,
        gate.min_cell_probability,
        "cell_probability",
    )
    if min_cell is not None:
        return min_cell
    return (
        _feature_max_reason(
            feature.centroid_distance,
            gate.max_centroid_distance,
            "centroid_distance",
        )
        or "accepted"
    )


def _teacher_edge_action_filter_reason(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    seed_session: int,
    action_filter: TeacherActionFilter | str,
) -> str:
    """Return a rejection reason when ``edge`` is not in the requested action class."""

    normalized = _normalize_teacher_action_filter(action_filter)
    if normalized == "all":
        return "accepted"
    action = _teacher_edge_action(predicted, edge, seed_session=seed_session)
    if normalized == "source-backfill" and action in {
        "source-backfill",
        "seed-source-backfill",
    }:
        return "accepted"
    if normalized == "completing-rescue":
        return (
            "accepted"
            if _teacher_edge_would_complete_row(predicted, edge)
            else "action_filter_completing-rescue"
        )
    if normalized == "completing-seed-source-backfill":
        return (
            "accepted"
            if action == "seed-source-backfill"
            and _teacher_edge_would_complete_row(predicted, edge)
            else "action_filter_completing-seed-source-backfill"
        )
    if normalized == "target-extension-or-seed-source-backfill" and action in {
        "target-extension",
        "seed-source-backfill",
    }:
        return "accepted"
    if action == normalized:
        return "accepted"
    return f"action_filter_{normalized}"


def _teacher_edge_action(
    predicted: np.ndarray, edge: TrackEdge, *, seed_session: int
) -> str:
    session_a, session_b, roi_a, roi_b = edge
    if int(session_b) != int(session_a) + 1:
        return "other"
    source_rows = tuple(np.flatnonzero(predicted[:, session_a] == roi_a))
    target_rows = tuple(np.flatnonzero(predicted[:, session_b] == roi_b))
    if len(source_rows) == 1 and len(target_rows) == 0:
        return "target-extension"
    if len(source_rows) == 0 and len(target_rows) == 1:
        return (
            "seed-source-backfill"
            if int(session_a) == int(seed_session)
            else "source-backfill"
        )
    if len(source_rows) == 1 and len(target_rows) == 1:
        return (
            "fragment-merge" if int(source_rows[0]) != int(target_rows[0]) else "other"
        )
    return "other"


def _teacher_edge_would_complete_row(predicted: np.ndarray, edge: TrackEdge) -> bool:
    """Return whether applying ``edge`` would make an affected row complete.

    This helper is intentionally label-free: it only inspects the current
    predicted matrix and the teacher edge endpoints. The normal rescue gates still
    decide whether the completion is allowed, teacher-supported, seed-anchored, and
    conflict-free. The action filter merely keeps small edit budgets focused on
    official complete-row opportunities instead of non-completing insertions.
    """

    session_a, session_b, roi_a, roi_b = edge
    if int(session_b) != int(session_a) + 1:
        return False
    source_rows = tuple(np.flatnonzero(predicted[:, session_a] == roi_a))
    target_rows = tuple(np.flatnonzero(predicted[:, session_b] == roi_b))
    if len(source_rows) == 1 and len(target_rows) == 0:
        candidate = predicted[int(source_rows[0])].copy()
        candidate[session_b] = roi_b
        return bool(np.all(candidate >= 0))
    if len(source_rows) == 0 and len(target_rows) == 1:
        candidate = predicted[int(target_rows[0])].copy()
        candidate[session_a] = roi_a
        return bool(np.all(candidate >= 0))
    if len(source_rows) == 1 and len(target_rows) == 1:
        source_row = int(source_rows[0])
        target_row = int(target_rows[0])
        if source_row == target_row:
            return False
        merged = _merge_rows_if_compatible(predicted[source_row], predicted[target_row])
        return bool(merged is not None and np.all(merged >= 0))
    return False


def _normalize_teacher_action_filter(
    action_filter: TeacherActionFilter | str,
) -> TeacherActionFilter:
    normalized = str(action_filter).strip().lower().replace("_", "-")
    allowed = {
        "all",
        "target-extension",
        "source-backfill",
        "seed-source-backfill",
        "fragment-merge",
        "target-extension-or-seed-source-backfill",
        "completing-rescue",
        "completing-seed-source-backfill",
    }
    if normalized not in allowed:
        raise ValueError(f"Unsupported teacher action filter: {action_filter!r}")
    return cast(TeacherActionFilter, normalized)


def _feature_min_reason(value: float, threshold: float | None, name: str) -> str | None:
    if threshold is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < float(threshold):
        return f"feature_gate_{name}"
    return None


def _feature_min_pair_reason(
    value_a: float, value_b: float, threshold: float | None, name: str
) -> str | None:
    if threshold is None:
        return None
    for value in (value_a, value_b):
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < float(threshold):
            return f"feature_gate_{name}"
    return None


def _feature_max_reason(value: float, threshold: float | None, name: str) -> str | None:
    if threshold is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or numeric > float(threshold):
        return f"feature_gate_{name}"
    return None


def _teacher_edge_rejection_row(edge: TrackEdge, reason: str) -> dict[str, int | str]:
    session_a, session_b, roi_a, roi_b = edge
    return {
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "applied": 0,
        "reason": reason,
        "source_row": -1,
        "target_row": -1,
        "teacher_complete_row_supported": 0,
    }


def _score_optional_float(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _try_apply_teacher_edge(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    seed_session: int,
    allow_completing_rescue: bool = False,
    allow_teacher_supported_completing_rescue: bool = False,
    teacher_complete_tracks: frozenset[tuple[int, ...]] = frozenset(),
    allow_completing_source_backfill: bool = False,
    allow_completing_fragment_merges: bool = False,
    allow_source_backfill: bool = True,
    allow_source_inserts: bool | None = None,
    allow_source_insertions: bool | None = None,
    allow_seed_source_backfill: bool = False,
    allow_completing_seed_source_backfill: bool = False,
    allow_fragment_merges: bool = True,
    min_component_observations: int = 1,
) -> tuple[np.ndarray, dict[str, int | str]]:
    allow_completing_rescue = _bool_value(
        allow_completing_rescue, name="allow_completing_rescue"
    )
    allow_teacher_supported_completing_rescue = _bool_value(
        allow_teacher_supported_completing_rescue,
        name="allow_teacher_supported_completing_rescue",
    )
    allow_completing_source_backfill = _bool_value(
        allow_completing_source_backfill,
        name="allow_completing_source_backfill",
    )
    allow_completing_fragment_merges = _bool_value(
        allow_completing_fragment_merges,
        name="allow_completing_fragment_merges",
    )
    allow_source_backfill = _bool_value(
        allow_source_backfill, name="allow_source_backfill"
    )
    allow_source_inserts = _optional_bool_value(
        allow_source_inserts, name="allow_source_inserts"
    )
    allow_source_insertions = _optional_bool_value(
        allow_source_insertions, name="allow_source_insertions"
    )
    allow_seed_source_backfill = _bool_value(
        allow_seed_source_backfill, name="allow_seed_source_backfill"
    )
    allow_completing_seed_source_backfill = _bool_value(
        allow_completing_seed_source_backfill,
        name="allow_completing_seed_source_backfill",
    )
    allow_fragment_merges = _bool_value(
        allow_fragment_merges, name="allow_fragment_merges"
    )
    output = np.asarray(predicted, dtype=int).copy()
    source_backfill_enabled = _resolve_source_backfill_alias(
        allow_source_backfill, allow_source_inserts, allow_source_insertions
    )
    session_a, session_b, roi_a, roi_b = edge
    row = {
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "applied": 0,
        "reason": "not_evaluated",
        "source_row": -1,
        "target_row": -1,
        "teacher_complete_row_supported": 0,
    }
    if session_b != session_a + 1:
        row["reason"] = "not_adjacent"
        return output, row
    source_rows = tuple(np.flatnonzero(output[:, session_a] == roi_a))
    target_rows = tuple(np.flatnonzero(output[:, session_b] == roi_b))
    if len(source_rows) > 1:
        row["reason"] = "missing_or_ambiguous_source"
        return output, row
    if len(target_rows) > 1:
        row["reason"] = "ambiguous_target_rows"
        return output, row

    if len(source_rows) == 1:
        source_row = int(source_rows[0])
        row["source_row"] = source_row
        if (
            output[source_row, session_b] >= 0
            and output[source_row, session_b] != roi_b
        ):
            row["reason"] = "source_has_target_conflict"
            return output, row
    else:
        source_row = -1

    if len(target_rows) == 1:
        target_row = int(target_rows[0])
        row["target_row"] = target_row
        if (
            output[target_row, session_a] >= 0
            and output[target_row, session_a] != roi_a
        ):
            row["reason"] = "target_has_source_conflict"
            return output, row
    else:
        target_row = -1

    if source_row >= 0 and target_row < 0:
        if not _row_is_seed_anchored(output[source_row], seed_session):
            row["reason"] = "source_not_seed_anchored"
            return output, row
        if (
            _component_observation_count(output[source_row])
            < min_component_observations
        ):
            row["reason"] = "insufficient_component_support"
            return output, row
        candidate_row = output[source_row].copy()
        candidate_row[session_b] = roi_b
        row["teacher_complete_row_supported"] = int(
            _teacher_complete_row_supported(candidate_row, teacher_complete_tracks)
        )
        if _would_complete_track(
            candidate_row,
            allow_completing_rescue,
            allow_teacher_supported_completing_rescue=(
                allow_teacher_supported_completing_rescue
            ),
            teacher_complete_tracks=teacher_complete_tracks,
        ):
            row["reason"] = "would_complete_track"
            return output, row
        output[source_row, session_b] = roi_b
        row["applied"] = 1
        row["reason"] = "accepted_insert_target"
        return output, row

    if source_row < 0 and target_row >= 0:
        target_seed_anchored = _row_is_seed_anchored(output[target_row], seed_session)
        seed_source_backfill = bool(
            allow_seed_source_backfill and session_a == seed_session
        )
        if not _source_backfill_allowed_for_edge(
            source_backfill_enabled,
            allow_seed_source_backfill=allow_seed_source_backfill,
            session_a=int(session_a),
            seed_session=int(seed_session),
        ):
            row["reason"] = "missing_or_ambiguous_source"
            return output, row
        candidate_row = output[target_row].copy()
        candidate_row[session_a] = roi_a
        teacher_completion_supported = _teacher_complete_row_supported(
            candidate_row, teacher_complete_tracks
        )
        row["teacher_complete_row_supported"] = int(teacher_completion_supported)
        if (
            not target_seed_anchored
            and not seed_source_backfill
            and not (
                allow_teacher_supported_completing_rescue
                and teacher_completion_supported
            )
        ):
            row["reason"] = "target_not_seed_anchored"
            return output, row
        if (
            _component_observation_count(output[target_row])
            < min_component_observations
        ):
            row["reason"] = "insufficient_component_support"
            return output, row
        allow_completion = bool(
            allow_completing_rescue
            or allow_completing_source_backfill
            or (seed_source_backfill and allow_completing_seed_source_backfill)
        )
        if _would_complete_track(
            candidate_row,
            allow_completion,
            allow_teacher_supported_completing_rescue=(
                allow_teacher_supported_completing_rescue
            ),
            teacher_complete_tracks=teacher_complete_tracks,
        ):
            row["reason"] = "would_complete_track"
            return output, row
        output[target_row, session_a] = roi_a
        row["applied"] = 1
        row["reason"] = (
            "accepted_merge_fragments"
            if (
                allow_teacher_supported_completing_rescue
                and teacher_completion_supported
                and not target_seed_anchored
            )
            else "accepted_insert_source"
        )
        return output, row

    if source_row >= 0 and target_row >= 0:
        if source_row == target_row:
            row["reason"] = "already_same_component"
            return output, row
        if not allow_fragment_merges:
            row["reason"] = "target_already_claimed"
            return output, row
        merged = _merge_rows_if_compatible(output[source_row], output[target_row])
        if merged is None:
            row["reason"] = "merge_conflict"
            return output, row
        if not _row_is_seed_anchored(merged, seed_session):
            row["reason"] = "merged_not_seed_anchored"
            return output, row
        source_support = _component_observation_count(output[source_row])
        target_support = _component_observation_count(output[target_row])
        if max(source_support, target_support) < min_component_observations:
            row["reason"] = "insufficient_component_support"
            return output, row
        row["teacher_complete_row_supported"] = int(
            _teacher_complete_row_supported(merged, teacher_complete_tracks)
        )
        if _would_complete_track(
            merged,
            bool(allow_completing_rescue or allow_completing_fragment_merges),
            allow_teacher_supported_completing_rescue=(
                allow_teacher_supported_completing_rescue
            ),
            teacher_complete_tracks=teacher_complete_tracks,
        ):
            row["reason"] = "would_complete_track"
            return output, row
        output[source_row] = merged
        output = np.delete(output, target_row, axis=0)
        row["applied"] = 1
        row["reason"] = "accepted_merge_fragments"
        return output, row

    row["reason"] = "missing_or_ambiguous_source"
    return output, row


def _row_is_seed_anchored(track_row: np.ndarray, seed_session: int) -> bool:
    return 0 <= seed_session < track_row.shape[0] and int(track_row[seed_session]) >= 0


def _component_observation_count(track_row: np.ndarray) -> int:
    return int(np.count_nonzero(track_row >= 0))


def _teacher_complete_row_supported(
    track_row: np.ndarray, teacher_complete_tracks: frozenset[tuple[int, ...]]
) -> bool:
    if not np.all(track_row >= 0):
        return False
    candidate = _row_tuple(track_row)
    return any(
        all(
            teacher_value < 0 or teacher_value == candidate[index]
            for index, teacher_value in enumerate(teacher_row)
        )
        for teacher_row in teacher_complete_tracks
    )


def _would_complete_track(
    track_row: np.ndarray,
    allow_completing_rescue: bool,
    *,
    allow_teacher_supported_completing_rescue: bool = False,
    teacher_complete_tracks: frozenset[tuple[int, ...]] = frozenset(),
) -> bool:
    if not np.all(track_row >= 0):
        return False
    if allow_completing_rescue:
        return False
    return not (
        allow_teacher_supported_completing_rescue
        and _teacher_complete_row_supported(track_row, teacher_complete_tracks)
    )


def _teacher_row_set(track_matrix: np.ndarray) -> frozenset[tuple[int, ...]]:
    return frozenset(_row_tuple(row) for row in track_matrix)


def _complete_row_set(track_matrix: np.ndarray) -> frozenset[tuple[int, ...]]:
    return frozenset(_row_tuple(row) for row in track_matrix if np.all(row >= 0))


def _row_tuple(track_row: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in track_row)


def _merge_rows_if_compatible(left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    conflict = (left >= 0) & (right >= 0) & (left != right)
    if np.any(conflict):
        return None
    return np.where(left >= 0, left, right)


def write_rescue_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-teacher-adjacent-rescue",
        description=(
            "Run Track2pPolicy component cleanup and then extend seed-anchored "
            "components with conflict-free adjacent Track2p teacher edges."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        default="manual-gt",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", choices=("auto", "suite2p", "npy"), default="suite2p"
    )
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    parser.add_argument(
        "--iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument(
        "--allow-completing-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow teacher rescue edges that would turn an incomplete "
            "seed-anchored component into a complete row."
        ),
    )
    parser.add_argument(
        "--allow-completing-fragment-merges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow only compatible fragment merges to produce a complete row. "
            "Target/source insertions that would complete a row remain disabled "
            "unless --allow-completing-rescue is also set."
        ),
    )
    parser.add_argument(
        "--allow-completing-source-backfill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow source-backfill teacher rescue edits to complete a row "
            "without also allowing plain target-extension completions."
        ),
    )
    parser.add_argument(
        "--allow-completing-fragment-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Compatibility alias for --allow-completing-fragment-merges."),
    )
    parser.add_argument(
        "--allow-teacher-complete-row-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Compatibility alias for --allow-teacher-supported-completing-rescue."),
    )
    parser.add_argument(
        "--allow-teacher-supported-completion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Compatibility alias for --allow-teacher-supported-completing-rescue."),
    )
    parser.add_argument(
        "--allow-teacher-supported-completing-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow a teacher rescue edit that completes a row only when the "
            "completed row is itself present as a complete Track2p teacher row. "
            "This is stricter than --allow-completing-rescue."
        ),
    )
    parser.add_argument(
        "--allow-teacher-confirmed-completing-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Compatibility alias for --allow-teacher-supported-completing-rescue."),
    )
    parser.add_argument(
        "--allow-source-backfill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow adjacent Track2p teacher edges to fill a missing source "
            "observation when the target observation is already in a "
            "seed-anchored component. Disable this while enabling "
            "--allow-seed-source-backfill to test only the missing-seed "
            "source-backfill bucket."
        ),
    )
    parser.add_argument(
        "--allow-source-inserts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compatibility alias for --allow-source-backfill. When set, it "
            "controls whether adjacent teacher edges can backfill an unclaimed "
            "source ROI into a component that already contains the target."
        ),
    )
    parser.add_argument(
        "--allow-source-insertions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compatibility alias for --allow-source-backfill. When set, it "
            "controls whether adjacent teacher edges can fill a missing source "
            "ROI into a compatible target-side component."
        ),
    )
    parser.add_argument(
        "--allow-seed-source-backfill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also allow source backfill when the missing source observation is "
            "the seed-session ROI, even when broad source backfill is disabled "
            "with --no-allow-source-backfill. This directly targets missing-seed residual "
            "errors, so it is opt-in."
        ),
    )
    parser.add_argument(
        "--allow-seed-completing-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Compatibility alias for --allow-completing-seed-source-backfill."),
    )
    parser.add_argument(
        "--allow-seed-completing-backfill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Compatibility alias for --allow-completing-seed-source-backfill."),
    )
    parser.add_argument(
        "--allow-completing-seed-source-backfill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow a seed-session source backfill to complete a component. This "
            "is narrower than --allow-completing-rescue and targets missing "
            "seed-session residual errors without allowing arbitrary completed "
            "teacher extensions."
        ),
    )
    parser.add_argument(
        "--allow-fragment-merges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow adjacent Track2p teacher edges to merge two compatible "
            "seed-anchored fragments instead of only inserting unclaimed targets."
        ),
    )
    parser.add_argument(
        "--min-component-observations",
        type=_positive_int_arg,
        default=1,
        help=(
            "Require the component receiving a teacher rescue edge to already "
            "contain at least this many observations."
        ),
    )
    parser.add_argument(
        "--max-applied-edits",
        type=_nonnegative_int_arg,
        default=None,
        help=(
            "Cap accepted teacher-rescue edits per subject. This tests the "
            "highest-priority few-edit regime without admitting the full teacher "
            "edge tail. Omit for no cap."
        ),
    )
    parser.add_argument(
        "--max-target-extension-edits",
        type=_nonnegative_int_arg,
        default=None,
        help="Cap accepted target-extension teacher edits per subject.",
    )
    parser.add_argument(
        "--max-source-backfill-edits",
        type=_nonnegative_int_arg,
        default=None,
        help="Cap accepted non-seed source-backfill teacher edits per subject.",
    )
    parser.add_argument(
        "--max-seed-source-backfill-edits",
        type=_nonnegative_int_arg,
        default=None,
        help="Cap accepted seed-session source-backfill teacher edits per subject.",
    )
    parser.add_argument(
        "--max-fragment-merge-edits",
        type=_nonnegative_int_arg,
        default=None,
        help="Cap accepted fragment-merge teacher edits per subject.",
    )
    parser.add_argument(
        "--max-completing-rescue-edits",
        type=_nonnegative_int_arg,
        default=None,
        help="Cap accepted teacher edits that would complete a predicted row.",
    )
    parser.add_argument(
        "--teacher-repair-preset",
        choices=(
            "none",
            "missing-seed-high-confidence",
            "missing-seed-cell-confident",
            "missing-seed-moderate-iou",
            "missing-seed-completing-cell-confident",
            "missing-seed-completing-moderate-iou",
            "track2p-fn-high-confidence",
            "track2p-fn-moderate-iou-cell-confident",
            "track2p-fn-moderate-iou-cell-confidence",
            "residual-union-cell-confident",
            "residual-union-action-specific",
            "residual-union-action-balanced",
            "residual-union-balanced",
            "completing-rescue-action-specific",
        ),
        default="none",
        help=(
            "Apply a narrow label-free teacher-rescue macro. "
            "'missing-seed-high-confidence' prioritizes seed-session source "
            "backfills with dynamic seed/confidence ordering, a small edit cap, "
            "and the seed-source-high-confidence feature gate; "
            "'missing-seed-cell-confident' targets the same action bucket with "
            "a stricter endpoint-cell gate, no positive-IoU floor, and a "
            "three-edit cap; "
            "'missing-seed-moderate-iou' adds the same action restriction "
            "with a moderate-IoU cell-confidence gate; "
            "'missing-seed-completing-moderate-iou' is stricter still: it "
            "admits only seed-source backfills that would immediately complete "
            "the predicted row, with dynamic seed-cell ordering and a one-edit "
            "cap; "
            "'track2p-fn-high-confidence' restricts rescue to high-confidence "
            "target extensions; 'track2p-fn-moderate-iou-cell-confident' tests "
            "the same residual Track2p-FN target-extension bucket with a "
            "moderate-IoU, cell-confident feature gate and cell-confidence "
            "dynamic ordering; "
            "'residual-union-cell-confident' combines target extensions and "
            "seed-source backfills with a cell-confident residual-FN feature "
            "gate, while disabling broad source backfill and fragment merges; "
            "'residual-union-action-specific' uses the same residual-union action "
            "filter but applies a stricter moderate-IoU/cell gate to target "
            "extensions and a seed-source cell-confident gate to missing-seed "
            "backfills. "
            "'residual-union-action-balanced' uses the same action-specific "
            "feature gates but reserves the tiny edit budget across residual "
            "families with per-action caps, so missing-seed backfills cannot "
            "crowd out Track2p-FN target extensions or vice versa. "
            "'completing-rescue-action-specific' spends a tiny budget only on "
            "teacher-confirmed adjacent edits that would complete a predicted "
            "row, with the same action-specific target-extension and seed-source "
            "feature gates, so non-completing teacher edits cannot consume the "
            "budget. "
            "Explicit non-default CLI values for order, feature preset, edit "
            "cap, and component support are preserved."
        ),
    )
    parser.add_argument(
        "--teacher-edge-order",
        choices=(
            "lexicographic",
            "structural",
            "dynamic-structural",
            "confidence",
            "cell-confidence",
            "dynamic-confidence",
            "dynamic-cell-confidence",
            "dynamic-seed-confidence",
            "dynamic-seed-cell-confidence",
        ),
        default="structural",
        help=(
            "Order Track2p teacher candidate edges lexicographically or by a "
            "label-free structural priority that favors merges/backfills first. "
            "dynamic-structural recomputes that priority after each attempted "
            "edit; confidence uses local registration evidence to break ties; "
            "cell-confidence orders by endpoint cell probability before raw "
            "IoU; dynamic-confidence does both; dynamic-seed-confidence "
            "additionally prioritizes missing seed-source backfills before "
            "other teacher edits; dynamic-cell-confidence and "
            "dynamic-seed-cell-confidence combine those behaviors."
        ),
    )
    parser.add_argument(
        "--teacher-action-filter",
        choices=(
            "all",
            "target-extension",
            "source-backfill",
            "seed-source-backfill",
            "fragment-merge",
            "target-extension-or-seed-source-backfill",
            "completing-rescue",
            "completing-seed-source-backfill",
        ),
        default="all",
        help=(
            "Restrict teacher rescue attempts to one structural action class. "
            "Use seed-source-backfill to target the residual missing-seed "
            "bucket, or target-extension-or-seed-source-backfill to test the "
            "two residual repair buckets without enabling broad source backfill "
            "or merges."
        ),
    )
    parser.add_argument(
        "--teacher-feature-preset",
        choices=(
            "none",
            "local-support",
            "high-confidence",
            "cell-high-confidence",
            "cell-confident",
            "track2p-fn-rescue",
            "residual-fn",
            "residual-fn-cell-confident",
            "moderate-iou-cell-confidence",
            "seed-source-high-confidence",
            "seed-source-cell-confident",
            "seed-source-moderate-iou",
        ),
        default="none",
        help=(
            "Apply a label-free feature-gate preset to teacher rescue edges. "
            "Explicit --teacher-* gate thresholds override the corresponding "
            "preset values. The cell-confident preset is the high-confidence "
            "gate plus a minimum Suite2p cell probability of 0.80 at both "
            "teacher-edge endpoints. residual-fn-cell-confident keeps the "
            "permissive residual-FN geometry gate but adds the same 0.80 "
            "endpoint cell-probability requirement and does not require a "
            "Hungarian assignment; seed-source-cell-confident is a missing-seed "
            "source-backfill gate with a stronger 0.85 endpoint cell-probability "
            "requirement and no positive-IoU floor; seed-source-moderate-iou "
            "additionally caps registered IoU to avoid high-overlap continuation spam."
        ),
    )
    parser.add_argument(
        "--target-extension-feature-preset",
        choices=(
            "none",
            "local-support",
            "high-confidence",
            "cell-high-confidence",
            "cell-confident",
            "track2p-fn-rescue",
            "residual-fn",
            "residual-fn-cell-confident",
            "moderate-iou-cell-confidence",
        ),
        default="none",
        help=(
            "Optional action-specific feature gate for target-extension teacher "
            "edits. When set, this gate overrides --teacher-feature-preset only "
            "for target extensions."
        ),
    )
    parser.add_argument(
        "--seed-source-feature-preset",
        choices=(
            "none",
            "seed-source-high-confidence",
            "seed-source-cell-confident",
            "seed-source-moderate-iou",
        ),
        default="none",
        help=(
            "Optional action-specific feature gate for seed-source-backfill "
            "teacher edits. When set, this gate overrides --teacher-feature-preset "
            "only for missing-seed source backfills."
        ),
    )
    parser.add_argument(
        "--teacher-min-registered-iou",
        "--teacher-gate-min-registered-iou",
        dest="teacher_min_registered_iou",
        type=float,
        default=None,
        help="Reject teacher rescue edges with registered IoU below this value.",
    )
    parser.add_argument(
        "--teacher-max-registered-iou",
        "--teacher-gate-max-registered-iou",
        dest="teacher_max_registered_iou",
        type=float,
        default=None,
        help="Reject teacher rescue edges with registered IoU above this value.",
    )
    parser.add_argument(
        "--teacher-min-threshold-margin",
        "--teacher-gate-min-threshold-margin",
        dest="teacher_min_threshold_margin",
        type=float,
        default=None,
        help="Reject teacher rescue edges below this threshold-margin value.",
    )
    parser.add_argument(
        "--teacher-min-row-margin",
        "--teacher-gate-min-row-margin",
        dest="teacher_min_row_margin",
        type=float,
        default=None,
        help="Reject teacher rescue edges below this row-margin value.",
    )
    parser.add_argument(
        "--teacher-min-column-margin",
        "--teacher-gate-min-column-margin",
        dest="teacher_min_column_margin",
        type=float,
        default=None,
        help="Reject teacher rescue edges below this column-margin value.",
    )
    parser.add_argument(
        "--teacher-max-centroid-distance",
        "--teacher-gate-max-centroid-distance",
        dest="teacher_max_centroid_distance",
        type=float,
        default=None,
        help="Reject teacher rescue edges with centroid distance above this value.",
    )
    parser.add_argument(
        "--teacher-min-area-ratio",
        "--teacher-gate-min-area-ratio",
        dest="teacher_min_area_ratio",
        type=float,
        default=None,
        help="Reject teacher rescue edges with ROI area ratio below this value.",
    )
    parser.add_argument(
        "--teacher-min-cell-probability",
        "--teacher-gate-min-cell-probability",
        dest="teacher_min_cell_probability",
        type=float,
        default=None,
        help=(
            "Reject teacher rescue edges when either endpoint has Suite2p cell "
            "probability below this value."
        ),
    )
    parser.add_argument(
        "--teacher-require-hungarian",
        "--teacher-require-hungarian-assignment",
        "--teacher-gate-require-hungarian",
        dest="teacher_require_hungarian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require teacher rescue edges to be assigned by the local Hungarian "
            "matcher before they can claim a slot."
        ),
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--diagnostics-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        plane_name=args.plane_name,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    preset_teacher_feature_gate = teacher_feature_gate_from_preset(
        cast(TeacherFeaturePreset, args.teacher_feature_preset)
    )
    manual_teacher_feature_gate = TeacherEdgeFeatureGate(
        min_registered_iou=args.teacher_min_registered_iou,
        max_registered_iou=args.teacher_max_registered_iou,
        min_threshold_margin=args.teacher_min_threshold_margin,
        min_row_margin=args.teacher_min_row_margin,
        min_column_margin=args.teacher_min_column_margin,
        max_centroid_distance=args.teacher_max_centroid_distance,
        min_area_ratio=args.teacher_min_area_ratio,
        min_cell_probability=args.teacher_min_cell_probability,
        require_hungarian=args.teacher_require_hungarian,
    )
    teacher_feature_gate = merge_teacher_feature_gates(
        preset_teacher_feature_gate, manual_teacher_feature_gate
    )
    output = run_track2p_policy_teacher_adjacent_rescue(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
        cleanup_config=cleanup_config,
        allow_completing_rescue=args.allow_completing_rescue,
        allow_teacher_complete_row_rescue=args.allow_teacher_complete_row_rescue,
        allow_teacher_supported_completion=args.allow_teacher_supported_completion,
        allow_teacher_supported_completing_rescue=(
            args.allow_teacher_supported_completing_rescue
        ),
        allow_teacher_confirmed_completing_rescue=(
            args.allow_teacher_confirmed_completing_rescue
        ),
        allow_completing_source_backfill=args.allow_completing_source_backfill,
        allow_completing_fragment_merge=args.allow_completing_fragment_merge,
        allow_completing_fragment_merges=args.allow_completing_fragment_merges,
        allow_source_backfill=args.allow_source_backfill,
        allow_source_inserts=args.allow_source_inserts,
        allow_source_insertions=args.allow_source_insertions,
        allow_seed_source_backfill=args.allow_seed_source_backfill,
        allow_seed_completing_backfill=args.allow_seed_completing_backfill,
        allow_seed_completing_rescue=args.allow_seed_completing_rescue,
        allow_completing_seed_source_backfill=(
            args.allow_completing_seed_source_backfill
        ),
        allow_fragment_merges=args.allow_fragment_merges,
        teacher_edge_order=cast(TeacherEdgeOrder, args.teacher_edge_order),
        teacher_action_filter=cast(TeacherActionFilter, args.teacher_action_filter),
        min_component_observations=args.min_component_observations,
        max_applied_edits=args.max_applied_edits,
        max_target_extension_edits=args.max_target_extension_edits,
        max_source_backfill_edits=args.max_source_backfill_edits,
        max_seed_source_backfill_edits=args.max_seed_source_backfill_edits,
        max_fragment_merge_edits=args.max_fragment_merge_edits,
        max_completing_rescue_edits=args.max_completing_rescue_edits,
        teacher_feature_gate=teacher_feature_gate,
        teacher_repair_preset=args.teacher_repair_preset,
        teacher_feature_preset=str(args.teacher_feature_preset),
        target_extension_feature_preset=str(args.target_extension_feature_preset),
        seed_source_feature_preset=str(args.seed_source_feature_preset),
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.diagnostics_output is not None:
        write_rescue_rows(
            output.component_rows,
            args.diagnostics_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
