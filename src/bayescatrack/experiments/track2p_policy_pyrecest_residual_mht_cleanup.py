"""PyRecEst-backed bounded residual MHT cleanup after CoherenceSuffixStitch.

This is a residual, edit-level MHT row, not a full all-ROI tracker.  It starts
from the non-teacher CoherenceSuffixStitch prediction, constructs label-free
growth-veto edit candidates, passes them to PyRecEst's generic residual-MHT
selector, applies the selected edit set, and reports the official benchmark
scores plus a candidate ledger.

Selection uses only label-free fields.  Manual-GT labels and score deltas can
appear in diagnostic rows inherited from growth-veto what-if machinery, but they
are never passed into the PyRecEst candidate score or conflict keys.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_stitch_whatif as suffix,
)
from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup
from bayescatrack.experiments import track2p_policy_growth_veto_whatif as veto
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)

try:
    from pyrecest.tracking import (
        ResidualEditCandidate,
        ResidualMHTConfig,
        enumerate_residual_hypotheses,
        select_residual_hypothesis,
    )
except ImportError as exc:  # pragma: no cover - exercised only with stale PyRecEst
    raise ImportError(
        "track2p-policy-pyrecest-residual-mht-cleanup requires a PyRecEst "
        "version that provides pyrecest.tracking.ResidualEditCandidate and "
        "select_residual_hypothesis. Apply the PyRecEst residual-edit MHT patch "
        "or install the corresponding PyRecEst branch."
    ) from exc

METHOD = "track2p-policy-pyrecest-residual-mht-cleanup"


@dataclass(frozen=True)
class PyRecEstResidualMHTOptions:
    """BayesCaTrack-side controls for the PyRecEst residual-MHT row."""

    candidate_top_k: int = 4
    max_edits_per_subject: int = 2
    max_hypotheses: int = 16
    edit_penalty: float = 0.25
    score_threshold: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_top_k",
            _positive_integer(self.candidate_top_k, name="candidate_top_k"),
        )
        object.__setattr__(
            self,
            "max_edits_per_subject",
            _nonnegative_integer(
                self.max_edits_per_subject,
                name="max_edits_per_subject",
            ),
        )
        object.__setattr__(
            self,
            "max_hypotheses",
            _positive_integer(self.max_hypotheses, name="max_hypotheses"),
        )
        object.__setattr__(
            self,
            "edit_penalty",
            _nonnegative_finite_float(self.edit_penalty, name="edit_penalty"),
        )
        object.__setattr__(
            self,
            "score_threshold",
            _validated_numeric_float(self.score_threshold, name="score_threshold"),
        )


@dataclass(frozen=True)
class PyRecEstResidualMHTResult:
    """Benchmark rows plus candidate/hypothesis diagnostics."""

    results: tuple[SubjectBenchmarkResult, ...]
    candidate_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


def run_track2p_policy_pyrecest_residual_mht_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    suffix_gate: suffix.CoherenceSuffixStitchGate | None = None,
    edge_top_k: int = 25,
    path_beam_width: int = 100,
    anchor_min_registered_iou: float = 0.50,
    anchor_min_shifted_iou: float = 0.30,
    anchor_min_cell_probability: float = 0.80,
    growth_veto_gate: cleanup.GrowthVetoGate | None = None,
    mht_options: PyRecEstResidualMHTOptions | None = None,
    progress: bool = False,
) -> PyRecEstResidualMHTResult:
    """Run PyRecEst residual-MHT cleanup from the non-teacher suffix row."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    suffix_gate = suffix_gate or suffix.CoherenceSuffixStitchGate()
    growth_veto_gate = growth_veto_gate or cleanup.GrowthVetoGate()
    mht_options = mht_options or PyRecEstResidualMHTOptions()

    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    states = [
        veto._subject_state(
            subject_dir,
            config=policy_config,
            cleanup_config=cleanup_config,
            suffix_gate=suffix_gate,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            edge_top_k=int(edge_top_k),
            path_beam_width=int(path_beam_width),
            anchor_min_registered_iou=float(anchor_min_registered_iou),
            anchor_min_shifted_iou=float(anchor_min_shifted_iou),
            anchor_min_cell_probability=float(anchor_min_cell_probability),
            prediction_base="coherence-suffix",
            progress=progress,
        )
        for subject_dir in subject_dirs
    ]
    global_baseline_scores = veto._global_scores(
        state.baseline_scores for state in states
    )

    results: list[SubjectBenchmarkResult] = []
    ledger_rows: list[dict[str, Any]] = []
    for state in states:
        edge_rows = veto._accepted_edge_rows(
            state,
            global_baseline_scores=global_baseline_scores,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            cell_probability_threshold=float(policy_config.cell_probability_threshold),
            transform_type=policy_config.transform_type,
        )
        edge_rows = cleanup._augment_growth_veto_candidate_shifted_iou(
            edge_rows,
            state.sessions,
            gate=growth_veto_gate,
            n_sessions=int(state.reference.shape[1]),
        )
        candidate_rows = _candidate_rows(
            edge_rows,
            gate=growth_veto_gate,
            options=mht_options,
            n_sessions=int(state.reference.shape[1]),
        )
        pyrecest_candidates = [
            _to_pyrecest_candidate(row, options=mht_options) for row in candidate_rows
        ]
        pyrecest_config = ResidualMHTConfig(
            max_edits=int(mht_options.max_edits_per_subject),
            max_hypotheses=int(mht_options.max_hypotheses),
            edit_penalty=float(mht_options.edit_penalty),
            score_threshold=float(mht_options.score_threshold),
            include_empty=True,
        )
        hypotheses = enumerate_residual_hypotheses(
            pyrecest_candidates,
            config=pyrecest_config,
        )
        selected_hypothesis = select_residual_hypothesis(
            pyrecest_candidates,
            config=pyrecest_config,
        )
        selected_ids = set(selected_hypothesis.candidate_ids)
        selected_rows = [
            row
            for row in candidate_rows
            if str(row["pyrecest_candidate_id"]) in selected_ids
        ]
        apply_gate = replace(
            growth_veto_gate,
            max_vetoes_per_subject=max(0, len(selected_rows)),
        )
        mht_tracks, applied_keys = cleanup._apply_growth_veto_rows(
            state.combined,
            selected_rows,
            gate=apply_gate,
        )
        scores = dict(score_track_matrices(mht_tracks, state.reference))
        scores.update(
            {
                "track2p_pyrecest_mht_candidates": int(len(candidate_rows)),
                "track2p_pyrecest_mht_hypotheses": int(len(hypotheses)),
                "track2p_pyrecest_mht_selected": int(len(selected_rows)),
                "track2p_pyrecest_mht_applied": int(len(applied_keys)),
                "track2p_pyrecest_mht_selected_score": float(selected_hypothesis.score),
                "track2p_pyrecest_mht_candidate_top_k": int(
                    mht_options.candidate_top_k
                ),
                "track2p_pyrecest_mht_max_edits_per_subject": int(
                    mht_options.max_edits_per_subject
                ),
                "track2p_pyrecest_mht_max_hypotheses": int(mht_options.max_hypotheses),
                "track2p_pyrecest_mht_edit_penalty": float(mht_options.edit_penalty),
                "track2p_pyrecest_mht_score_threshold": float(
                    mht_options.score_threshold
                ),
            }
        )
        results.append(
            SubjectBenchmarkResult(
                subject=state.subject,
                variant="CoherenceSuffixStitch + PyRecEst residual MHT",
                method=cast(Any, METHOD),
                scores=scores,
                n_sessions=int(state.reference.shape[1]),
                reference_source=GROUND_TRUTH_REFERENCE_SOURCE,
            )
        )

        selected_keys = {cleanup._edge_row_key(row) for row in selected_rows}
        applied_set = set(applied_keys)
        hypothesis_ids = [";".join(h.candidate_ids) for h in hypotheses]
        for row in edge_rows:
            key = cleanup._edge_row_key(row)
            candidate_id = _candidate_id(row)
            is_candidate = any(
                str(candidate_row["pyrecest_candidate_id"]) == candidate_id
                for candidate_row in candidate_rows
            )
            ledger_rows.append(
                {
                    **row,
                    "pyrecest_candidate_id": candidate_id,
                    "pyrecest_candidate": int(is_candidate),
                    "pyrecest_candidate_score": float(
                        _candidate_score(row, options=mht_options)
                    ),
                    "selected_by_pyrecest_mht": int(key in selected_keys),
                    "applied_by_pyrecest_mht": int(key in applied_set),
                    "pyrecest_selected_hypothesis": ";".join(
                        selected_hypothesis.candidate_ids
                    ),
                    "pyrecest_selected_hypothesis_score": float(
                        selected_hypothesis.score
                    ),
                    "pyrecest_hypothesis_count": int(len(hypotheses)),
                    "pyrecest_top_hypotheses": "|".join(hypothesis_ids[:5]),
                    "pyrecest_mht_gate_reason": cleanup.growth_veto_gate_reason(
                        row,
                        growth_veto_gate,
                        n_sessions=int(state.reference.shape[1]),
                    ),
                }
            )

    return PyRecEstResidualMHTResult(
        tuple(results),
        tuple(ledger_rows),
        tuple(_summary_rows(ledger_rows)),
    )


def _candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    gate: cleanup.GrowthVetoGate,
    options: PyRecEstResidualMHTOptions,
    n_sessions: int,
) -> list[Mapping[str, Any]]:
    candidates = [
        {**dict(row), "pyrecest_candidate_id": _candidate_id(row)}
        for row in rows
        if cleanup.growth_veto_gate_reason(row, gate, n_sessions=n_sessions)
        == "accepted"
    ]
    candidates.sort(
        key=lambda row: (
            -_candidate_score(row, options=options),
            str(row["pyrecest_candidate_id"]),
        )
    )
    return candidates[: int(options.candidate_top_k)]


def _to_pyrecest_candidate(
    row: Mapping[str, Any], *, options: PyRecEstResidualMHTOptions
) -> ResidualEditCandidate:
    return ResidualEditCandidate(
        candidate_id=str(row["pyrecest_candidate_id"]),
        score=float(_candidate_score(row, options=options)),
        conflict_keys=_conflict_keys(row),
        metadata={
            "subject": str(row.get("subject", "")),
            "session_a": int(row.get("session_a", -1)),
            "session_b": int(row.get("session_b", -1)),
            "roi_a": int(row.get("roi_a", -1)),
            "roi_b": int(row.get("roi_b", -1)),
        },
    )


def _candidate_id(row: Mapping[str, Any]) -> str:
    return ":".join(
        [
            str(row.get("subject", "")),
            str(int(row.get("session_a", -1))),
            str(int(row.get("session_b", -1))),
            str(int(row.get("roi_a", -1))),
            str(int(row.get("roi_b", -1))),
            str(int(row.get("occurrence_index", 0))),
        ]
    )


def _conflict_keys(row: Mapping[str, Any]) -> frozenset[str]:
    subject = str(row.get("subject", ""))
    return frozenset(
        {
            f"edge:{_candidate_id(row)}",
            f"target:{subject}:{int(row.get('session_b', -1))}:{int(row.get('roi_b', -1))}",
            f"source:{subject}:{int(row.get('session_a', -1))}:{int(row.get('roi_a', -1))}",
        }
    )


def _candidate_score(
    row: Mapping[str, Any], *, options: PyRecEstResidualMHTOptions
) -> float:
    """Label-free residual-MHT score; ignores all GT/status/delta columns."""

    del options
    growth_mahal = _finite_float(row.get("growth_residual_mahalanobis"), 0.0)
    growth_residual = _finite_float(row.get("growth_residual"), 0.0)
    registered_iou = _finite_float(row.get("registered_iou"), 0.0)
    shifted_iou = _finite_float(row.get("shifted_iou"), 0.0)
    local_distortion = _finite_float(row.get("local_neighbor_distortion"), 0.0)
    row_rank = max(1, int(_finite_float(row.get("row_rank"), 9.0)))
    column_rank = max(1, int(_finite_float(row.get("column_rank"), 9.0)))
    terminal = float(int(row.get("is_terminal_edge", 0)) > 0)
    last_session = float(int(row.get("is_last_session_edge", 0)) > 0)
    complete_component = float(int(row.get("complete_component_size", 0)) > 0)

    score = 0.0
    score += min(3.0, growth_mahal / 10.0)
    score += min(1.5, growth_residual / 3.0)
    score += 0.50 * shifted_iou
    score += 0.25 * registered_iou
    score += 0.20 * complete_component
    score += 0.15 * terminal
    score += 0.15 * last_session
    score -= 0.10 * float(row_rank - 1)
    score -= 0.10 * float(column_rank - 1)
    score -= 1.00 * max(0.0, local_distortion)
    return float(score)


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row.get("subject", ""))].append(row)
    by_subject["ALL"] = list(rows)
    output: list[dict[str, Any]] = []
    for subject, subject_rows in sorted(by_subject.items()):
        candidates = [
            row for row in subject_rows if int(row.get("pyrecest_candidate", 0))
        ]
        selected = [
            row for row in subject_rows if int(row.get("selected_by_pyrecest_mht", 0))
        ]
        applied = [
            row for row in subject_rows if int(row.get("applied_by_pyrecest_mht", 0))
        ]
        output.append(
            {
                "subject": subject,
                "accepted_edges": int(len(subject_rows)),
                "pyrecest_candidates": int(len(candidates)),
                "selected_by_pyrecest_mht": int(len(selected)),
                "applied_by_pyrecest_mht": int(len(applied)),
                "selected_true_positive_edges": int(
                    sum(
                        str(row.get("edge_status_against_gt")) == "true_positive"
                        for row in selected
                    )
                ),
                "selected_false_positive_edges": int(
                    sum(
                        str(row.get("edge_status_against_gt")) == "false_positive"
                        for row in selected
                    )
                ),
            }
        )
    return output


def _finite_float(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return numeric if np.isfinite(numeric) else float(fallback)


def _validated_numeric_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _nonnegative_finite_float(value: Any, *, name: str) -> float:
    numeric = _validated_numeric_float(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _positive_integer(value: Any, *, name: str) -> int:
    numeric = _validated_numeric_float(value, name=name)
    if not numeric.is_integer() or numeric < 1.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _nonnegative_integer(value: Any, *, name: str) -> int:
    numeric = _validated_numeric_float(value, name=name)
    if not numeric.is_integer() or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)


def _option_present(args: Sequence[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = cleanup.build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-policy-pyrecest-residual-mht-cleanup"
    parser.description = (
        "Run PyRecEst bounded residual MHT over label-free growth-veto hypotheses "
        "after CoherenceSuffixStitch."
    )
    parser.set_defaults(growth_veto_base="coherence-suffix")
    for action in parser._actions:
        if action.dest == "growth_veto_base":
            action.choices = ("coherence-suffix",)
            action.default = "coherence-suffix"
            action.help = (
                "Residual-MHT always starts from the non-teacher "
                "CoherenceSuffixStitch state."
            )
    parser.add_argument("--mht-candidate-top-k", type=int, default=4)
    parser.add_argument("--mht-max-edits-per-subject", type=int, default=2)
    parser.add_argument("--mht-max-hypotheses", type=int, default=16)
    parser.add_argument("--mht-edit-penalty", type=float, default=0.25)
    parser.add_argument("--mht-score-threshold", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(raw_args)
    if args.growth_veto_base != "coherence-suffix":
        parser.error(
            "PyRecEst residual MHT always starts from CoherenceSuffixStitch; "
            "use --growth-veto-base coherence-suffix."
        )
    mht_candidate_top_k = int(args.mht_candidate_top_k)
    mht_max_edits_per_subject = int(args.mht_max_edits_per_subject)
    legacy_veto_cap_present = any(
        _option_present(raw_args, option)
        for option in (
            "--max-vetoes-per-subject",
            "--growth-veto-max-vetoes-per-subject",
        )
    )
    if legacy_veto_cap_present:
        legacy_veto_cap = int(args.max_vetoes_per_subject)
        if not _option_present(raw_args, "--mht-candidate-top-k"):
            mht_candidate_top_k = legacy_veto_cap
        if not _option_present(raw_args, "--mht-max-edits-per-subject"):
            mht_max_edits_per_subject = legacy_veto_cap
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
        allow_track2p_as_reference_for_smoke_test=(
            args.allow_track2p_as_reference_for_smoke_test
        ),
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    suffix_gate = suffix.CoherenceSuffixStitchGate(
        suffix_path_length=int(args.suffix_path_length),
        min_cell_probability=float(args.min_cell_probability),
        min_area_ratio=float(args.min_area_ratio),
        max_centroid_distance=float(args.max_centroid_distance),
        min_shifted_iou=float(args.min_shifted_iou),
        min_motion_consistency=float(args.min_motion_consistency),
        min_shape_consistency=float(args.min_shape_consistency),
        max_stitches_per_subject=int(args.max_stitches_per_subject),
    )
    result = run_track2p_policy_pyrecest_residual_mht_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
        suffix_gate=suffix_gate,
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
        anchor_min_registered_iou=float(args.anchor_min_registered_iou),
        anchor_min_shifted_iou=float(args.anchor_min_shifted_iou),
        anchor_min_cell_probability=float(args.anchor_min_cell_probability),
        growth_veto_gate=cleanup.GrowthVetoGate(
            min_growth_residual_mahalanobis=float(args.min_growth_residual_mahalanobis),
            min_growth_residual=float(args.min_growth_residual),
            min_registered_iou=float(args.min_veto_registered_iou),
            min_shifted_iou=float(args.min_veto_shifted_iou),
            max_registered_iou=args.max_veto_registered_iou,
            max_shifted_iou=args.max_veto_shifted_iou,
            min_cell_probability=float(args.min_veto_cell_probability),
            max_min_cell_probability=(
                None
                if args.max_veto_min_cell_probability is None
                else float(args.max_veto_min_cell_probability)
            ),
            max_local_neighbor_distortion=(
                None
                if args.max_veto_local_neighbor_distortion is None
                else float(args.max_veto_local_neighbor_distortion)
            ),
            min_anchor_count=max(0, int(args.min_veto_anchor_count)),
            min_complete_component_size=(
                None
                if args.min_veto_complete_component_size is None
                else max(0, int(args.min_veto_complete_component_size))
            ),
            max_row_rank=int(args.max_veto_row_rank),
            max_column_rank=int(args.max_veto_column_rank),
            require_not_suffix_edge=bool(args.require_veto_not_suffix_edge),
            require_terminal_edge=bool(args.require_veto_terminal_edge),
            require_last_session_edge=bool(args.require_veto_last_session_edge),
            require_complete_component=bool(args.require_veto_complete_component),
            max_vetoes_per_subject=int(mht_candidate_top_k),
        ),
        mht_options=PyRecEstResidualMHTOptions(
            candidate_top_k=int(mht_candidate_top_k),
            max_edits_per_subject=int(mht_max_edits_per_subject),
            max_hypotheses=int(args.mht_max_hypotheses),
            edit_penalty=float(args.mht_edit_penalty),
            score_threshold=float(args.mht_score_threshold),
        ),
        progress=bool(args.progress),
    )
    write_results(
        [benchmark_result.to_dict() for benchmark_result in result.results],
        args.output,
        cast(OutputFormat, args.format),
    )
    if args.diagnostics_output is not None:
        veto.write_rows(
            result.candidate_rows,
            args.diagnostics_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    if args.candidate_output is not None:
        veto.write_rows(
            result.candidate_rows,
            args.candidate_output,
            output_format=cast(Literal["csv", "json"], args.format),
        )
    if args.summary_output is not None:
        veto.write_rows(
            result.summary_rows,
            args.summary_output,
            output_format=cast(Literal["csv", "json"], args.format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
