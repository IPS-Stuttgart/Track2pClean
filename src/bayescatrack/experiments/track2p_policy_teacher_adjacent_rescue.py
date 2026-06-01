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
stitch-only rescue. A compatibility seed-completing backfill opt-in allows only
Track2p-supported source backfills that fill the seed-session ROI. The command
does not use manual GT labels to choose edges.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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
    "dynamic-confidence",
]
TeacherFeaturePreset = Literal["none", "local-support", "high-confidence"]


@dataclass(frozen=True)
class TeacherAdjacentRescueReport:
    """Prediction plus teacher-rescue diagnostic rows."""

    tracks: np.ndarray
    rows: tuple[dict[str, int | str], ...]


@dataclass(frozen=True, init=False)
class TeacherEdgeFeatureGate:
    """Label-free local-evidence gate for Track2p teacher rescue edges."""

    min_registered_iou: float | None
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
            require_hungarian = bool(require_hungarian_assignment)
        if require_assigned_by_hungarian is not None:
            require_hungarian = bool(require_assigned_by_hungarian)
        object.__setattr__(self, "min_registered_iou", min_registered_iou)
        object.__setattr__(self, "min_threshold_margin", min_threshold_margin)
        object.__setattr__(self, "min_row_margin", min_row_margin)
        object.__setattr__(self, "min_column_margin", min_column_margin)
        object.__setattr__(self, "max_centroid_distance", max_centroid_distance)
        object.__setattr__(self, "min_area_ratio", min_area_ratio)
        object.__setattr__(self, "min_cell_probability", min_cell_probability)
        object.__setattr__(self, "require_hungarian", bool(require_hungarian))

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
            or self.min_threshold_margin is not None
            or self.min_row_margin is not None
            or self.min_column_margin is not None
            or self.max_centroid_distance is not None
            or self.min_area_ratio is not None
            or self.min_cell_probability is not None
        )


TeacherFeatureGate = TeacherEdgeFeatureGate


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
    raise ValueError(f"Unsupported teacher feature preset: {preset!r}")


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
        require_hungarian=manual_gate.require_hungarian or preset_gate.require_hungarian,
    )


def _resolve_source_backfill_alias(
    allow_source_backfill: bool,
    allow_source_inserts: bool | None,
    allow_source_insertions: bool | None = None,
) -> bool:
    if allow_source_insertions is not None:
        return bool(allow_source_insertions)
    if allow_source_inserts is None:
        return bool(allow_source_backfill)
    return bool(allow_source_inserts)


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
    min_component_observations: int = 1,
    max_applied_edits: int | None = None,
    teacher_feature_gate: TeacherEdgeFeatureGate | None = None,
    edge_feature_gate: TeacherEdgeFeatureGate | None = None,
    teacher_feature_preset: str = "none",
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
    source_backfill_enabled = _resolve_source_backfill_alias(
        allow_source_backfill, allow_source_inserts, allow_source_insertions
    )
    allow_teacher_supported_completion_enabled = bool(
        allow_teacher_complete_row_rescue
        or allow_teacher_supported_completion
        or allow_teacher_supported_completing_rescue
        or allow_teacher_confirmed_completing_rescue
    )
    allow_seed_completion = bool(
        allow_seed_completing_backfill
        or allow_seed_completing_rescue
        or allow_completing_seed_source_backfill
    )
    allow_fragment_completion = bool(
        allow_completing_fragment_merge or allow_completing_fragment_merges
    )
    min_component_observations = max(1, int(min_component_observations))
    teacher_feature_gate = _resolve_teacher_feature_gate(
        teacher_feature_gate, edge_feature_gate
    )
    teacher_feature_gate = merge_teacher_feature_gates(
        teacher_feature_gate_from_preset(teacher_feature_preset),
        teacher_feature_gate or TeacherEdgeFeatureGate(),
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
            if teacher_edge_order in {"confidence", "dynamic-confidence"}
            or _teacher_feature_gate_enabled(teacher_feature_gate)
            else {}
        )
        rescue = apply_teacher_adjacent_rescue_edges(
            base_full,
            teacher_full,
            seed_session=policy_config.seed_session,
            allow_completing_rescue=allow_completing_rescue,
            allow_teacher_supported_completing_rescue=(
                allow_teacher_supported_completion_enabled
            ),
            allow_completing_source_backfill=allow_completing_source_backfill,
            allow_completing_fragment_merges=allow_fragment_completion,
            allow_source_backfill=source_backfill_enabled,
            allow_seed_source_backfill=allow_seed_source_backfill,
            allow_completing_seed_source_backfill=(allow_seed_completion),
            allow_fragment_merges=allow_fragment_merges,
            edge_order=teacher_edge_order,
            edge_feature_index=edge_feature_index,
            teacher_feature_gate=teacher_feature_gate,
            min_component_observations=min_component_observations,
            max_applied_edits=max_applied_edits,
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
                allow_teacher_supported_completion_enabled
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
            "track2p_teacher_adjacent_min_component_observations": int(
                min_component_observations
            ),
            "track2p_teacher_adjacent_max_applied_edits": (
                -1 if max_applied_edits is None else int(max_applied_edits)
            ),
            "track2p_teacher_adjacent_feature_preset": str(
                teacher_feature_preset
            ),
            "track2p_teacher_adjacent_feature_gate_enabled": int(
                _teacher_feature_gate_enabled(teacher_feature_gate)
            ),
            "track2p_teacher_adjacent_min_registered_iou": _score_optional_float(
                None
                if teacher_feature_gate is None
                else teacher_feature_gate.min_registered_iou
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
    edge_feature_index: Mapping[TrackEdge, ResidualFeature] | None = None,
    teacher_feature_gate: TeacherEdgeFeatureGate | None = None,
    feature_gate: TeacherEdgeFeatureGate | None = None,
    edge_feature_gate: TeacherEdgeFeatureGate | None = None,
    min_component_observations: int = 1,
    max_applied_edits: int | None = None,
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
    ``min_component_observations`` is a label-free support gate: teacher edits
    must touch at least one component with this many existing observations.

    ``max_applied_edits`` caps the number of accepted teacher edits per subject.
    This makes it possible to test the high-confidence first-edit regime without
    admitting a long tail of Track2p-teacher edges after the best rescue.
    """

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
    min_component_observations = max(1, int(min_component_observations))
    max_applied_edits = _normalized_max_applied_edits(max_applied_edits)
    teacher_feature_gate = _resolve_teacher_feature_gate(
        teacher_feature_gate, feature_gate, edge_feature_gate
    )
    source_backfill_enabled = _resolve_source_backfill_alias(
        allow_source_backfill, allow_source_inserts, allow_source_insertions
    )
    if edge_order in {"dynamic-structural", "dynamic-confidence"}:
        return _apply_teacher_adjacent_rescue_edges_dynamic(
            output,
            teacher,
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
            edge_feature_index=edge_feature_index or {},
            use_confidence_order=edge_order == "dynamic-confidence",
            teacher_feature_gate=teacher_feature_gate,
            max_applied_edits=max_applied_edits,
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
        gate_reason = _teacher_edge_feature_gate_reason(
            (edge_feature_index or {}).get(edge), teacher_feature_gate
        )
        if gate_reason != "accepted":
            rows.append(
                {
                    **_teacher_edge_rejection_row(edge, gate_reason),
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
    teacher_feature_gate: TeacherEdgeFeatureGate | None,
    max_applied_edits: int | None,
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

    while True:
        output_counts = track_edge_counter(output)
        pending: list[tuple[TrackEdge, int]] = []
        for edge, occurrence_index in occurrences:
            occurrence = (edge, occurrence_index)
            if occurrence in attempted:
                continue
            if output_counts.get(edge, 0) > occurrence_index:
                attempted.add(occurrence)
                continue
            gate_reason = _teacher_edge_feature_gate_reason(
                edge_feature_index.get(edge), teacher_feature_gate
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
        rows.append({**row, "occurrence_index": int(occurrence_index)})

    return TeacherAdjacentRescueReport(output, tuple(rows))


def _normalized_max_applied_edits(max_applied_edits: int | None) -> int | None:
    if max_applied_edits is None:
        return None
    return max(0, int(max_applied_edits))


def _max_applied_edits_reached(
    applied_count: int, max_applied_edits: int | None
) -> bool:
    return max_applied_edits is not None and int(applied_count) >= int(
        max_applied_edits
    )


def _teacher_edge_limit_row(edge: TrackEdge) -> dict[str, int | str]:
    session_a, session_b, roi_a, roi_b = edge
    return {
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "applied": 0,
        "reason": "max_applied_edits_reached",
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
    if edge_order in {"dynamic-structural", "dynamic-confidence"}:
        return occurrences
    if edge_order == "confidence":
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
    current predicted matrix: compatible fragment merges outrank source
    backfills, which outrank plain target extensions. Within an action class,
    edits involving more already-supported observations are tried first.
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
                action_rank = 0
                evidence = int(np.count_nonzero(merged >= 0))
    elif source_row < 0 and target_row >= 0:
        target_seed_anchored = _row_is_seed_anchored(
            predicted[target_row], seed_session
        )
        seed_source_backfill = bool(
            allow_seed_source_backfill and session_a == seed_session
        )
        if allow_source_backfill and (
            target_seed_anchored
            or seed_source_backfill
            or allow_teacher_supported_completing_rescue
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
                action_rank = 1
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
                action_rank = 2
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
) -> tuple[int, int, int, float, float, float, float, float, float, int, int, int, int]:
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
    feature_key = _teacher_edge_feature_order_key(edge_feature_index.get(edge))
    return (
        structural[0],
        structural[1],
        *feature_key,
        structural[2],
        structural[3],
        structural[4],
        structural[5],
    )


def _teacher_edge_feature_order_key(
    feature: ResidualFeature | None,
) -> tuple[int, float, float, float, float, float, float, float]:
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
        -min_cell_probability,
        centroid_distance,
        -area_ratio,
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
        if not source_backfill_enabled:
            row["reason"] = "missing_or_ambiguous_source"
            return output, row
        target_seed_anchored = _row_is_seed_anchored(output[target_row], seed_session)
        seed_source_backfill = bool(
            allow_seed_source_backfill and session_a == seed_session
        )
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
            "seed-anchored component."
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
            "the seed-session ROI. This directly targets missing-seed residual "
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
        type=int,
        default=1,
        help=(
            "Require the component receiving a teacher rescue edge to already "
            "contain at least this many observations."
        ),
    )
    parser.add_argument(
        "--max-applied-edits",
        type=int,
        default=None,
        help=(
            "Cap accepted teacher-rescue edits per subject. This tests the "
            "highest-priority few-edit regime without admitting the full teacher "
            "edge tail. Omit for no cap."
        ),
    )
    parser.add_argument(
        "--teacher-edge-order",
        choices=(
            "lexicographic",
            "structural",
            "dynamic-structural",
            "confidence",
            "dynamic-confidence",
        ),
        default="structural",
        help=(
            "Order Track2p teacher candidate edges lexicographically or by a "
            "label-free structural priority that favors merges/backfills first. "
            "dynamic-structural recomputes that priority after each attempted "
            "edit; confidence uses local registration evidence to break ties; "
            "dynamic-confidence does both."
        ),
    )
    parser.add_argument(
        "--teacher-feature-preset",
        choices=("none", "local-support", "high-confidence"),
        default="none",
        help=(
            "Apply a label-free feature-gate preset to teacher rescue edges. "
            "Explicit --teacher-* gate thresholds override the corresponding "
            "preset values."
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
        min_component_observations=args.min_component_observations,
        max_applied_edits=args.max_applied_edits,
        teacher_feature_gate=teacher_feature_gate,
        teacher_feature_preset=str(args.teacher_feature_preset),
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
