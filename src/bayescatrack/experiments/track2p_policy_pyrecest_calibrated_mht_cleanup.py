"""LOSO-calibrated PyRecEst residual MHT cleanup after CoherenceSuffixStitch.

This experimental row replaces hand-tuned high-overlap residual pockets with a
single fold-calibrated false-positive probability for structurally valid edit
candidates.  Manual-GT labels are used only inside each training fold to fit the
calibrator and choose one probability threshold; held-out candidate exposure and
PyRecEst MHT scoring use only label-free residual/diagnostic features.
"""

from __future__ import annotations

import argparse
import math
import sys
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
from bayescatrack.experiments import (
    track2p_policy_pyrecest_residual_mht_cleanup as residual_mht,
)
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

METHOD = "track2p-policy-pyrecest-calibrated-mht-cleanup"
CALIBRATED_FEATURE_NAMES = (
    "registered_iou",
    "log1p_growth_residual",
    "log1p_growth_residual_mahalanobis",
    "min_endpoint_cell_probability",
)
_UNSUPPORTED_CALIBRATED_GROWTH_VETO_OPTIONS = frozenset(
    {
        "--min-growth-residual-mahalanobis",
        "--growth-veto-min-mahalanobis",
        "--min-growth-residual",
        "--growth-veto-min-residual",
        "--min-veto-registered-iou",
        "--growth-veto-min-registered-iou",
        "--max-veto-registered-iou",
        "--growth-veto-max-registered-iou",
        "--min-veto-shifted-iou",
        "--growth-veto-min-shifted-iou",
        "--max-veto-shifted-iou",
        "--growth-veto-max-shifted-iou",
        "--max-vetoes-per-subject",
        "--growth-veto-max-vetoes-per-subject",
    }
)


@dataclass(frozen=True)
class CalibratedResidualMHTOptions:
    """Controls for the fold-calibrated residual-MHT row."""

    max_edits_per_subject: int = 4
    max_hypotheses: int = 64
    edit_penalty: float = 0.0
    score_threshold: float = 0.0
    logistic_c: float = 0.5
    min_training_positive_examples: int = 1


@dataclass(frozen=True)
class FalsePositiveCalibrator:
    """Small, serializable logistic false-positive probability model."""

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def predict_false_positive_probability(self, row: Mapping[str, Any]) -> float:
        features = _calibrated_feature_vector(row)
        score = float(self.intercept)
        for value, mean, scale, coefficient in zip(
            features,
            self.feature_mean,
            self.feature_scale,
            self.coefficients,
            strict=True,
        ):
            score += float(coefficient) * ((float(value) - mean) / scale)
        return _sigmoid(score)


@dataclass(frozen=True)
class CalibratedResidualMHTResult:
    """Benchmark rows plus calibrated candidate diagnostics."""

    results: tuple[SubjectBenchmarkResult, ...]
    candidate_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _SubjectRows:
    subject: str
    n_sessions: int
    state: Any
    edge_rows: tuple[dict[str, Any], ...]


def run_track2p_policy_pyrecest_calibrated_mht_cleanup(
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
    structural_gate: cleanup.GrowthVetoGate | None = None,
    mht_options: CalibratedResidualMHTOptions | None = None,
    progress: bool = False,
) -> CalibratedResidualMHTResult:
    """Run LOSO-calibrated residual MHT from the non-teacher suffix row."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    suffix_gate = suffix_gate or suffix.CoherenceSuffixStitchGate()
    structural_gate = structural_gate or cleanup.GrowthVetoGate()
    mht_options = mht_options or CalibratedResidualMHTOptions()

    subject_dirs = discover_subject_dirs(policy_config.data)
    if len(subject_dirs) < 2:
        raise ValueError("Calibrated residual MHT requires at least two subjects")

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
    subject_rows = tuple(
        _edge_rows_for_state(
            state,
            global_baseline_scores=global_baseline_scores,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            cell_probability_threshold=float(policy_config.cell_probability_threshold),
            transform_type=policy_config.transform_type,
            structural_gate=structural_gate,
        )
        for state in states
    )

    results: list[SubjectBenchmarkResult] = []
    ledger_rows: list[dict[str, Any]] = []
    for held_out in subject_rows:
        training_structural_rows = [
            row
            for subject in subject_rows
            if subject.subject != held_out.subject
            for row in _structural_candidate_rows(
                subject.edge_rows,
                gate=structural_gate,
                n_sessions=subject.n_sessions,
            )
        ]
        calibrator = _fit_false_positive_calibrator(
            training_structural_rows,
            options=mht_options,
        )
        training_probabilities = [
            calibrator.predict_false_positive_probability(row)
            for row in training_structural_rows
        ]
        threshold = _select_training_probability_threshold(
            training_structural_rows,
            training_probabilities,
        )
        candidate_rows = _calibrated_candidate_rows(
            held_out.edge_rows,
            gate=structural_gate,
            n_sessions=held_out.n_sessions,
            calibrator=calibrator,
            threshold=threshold,
        )
        pyrecest_candidates = [
            _to_calibrated_pyrecest_candidate(row) for row in candidate_rows
        ]
        pyrecest_config = residual_mht.ResidualMHTConfig(
            max_edits=int(mht_options.max_edits_per_subject),
            max_hypotheses=int(mht_options.max_hypotheses),
            edit_penalty=float(mht_options.edit_penalty),
            score_threshold=float(mht_options.score_threshold),
            include_empty=True,
        )
        hypotheses = residual_mht.enumerate_residual_hypotheses(
            pyrecest_candidates,
            config=pyrecest_config,
        )
        selected_hypothesis = residual_mht.select_residual_hypothesis(
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
            structural_gate,
            max_vetoes_per_subject=max(0, len(selected_rows)),
        )
        mht_tracks, applied_keys = cleanup._apply_growth_veto_rows(
            held_out.state.combined,
            selected_rows,
            gate=apply_gate,
        )
        scores = dict(score_track_matrices(mht_tracks, held_out.state.reference))
        scores.update(
            {
                "track2p_pyrecest_calibrated_mht_training_examples": int(
                    len(training_structural_rows)
                ),
                "track2p_pyrecest_calibrated_mht_training_positive_examples": int(
                    sum(_false_positive_label(row) for row in training_structural_rows)
                ),
                "track2p_pyrecest_calibrated_mht_threshold": float(threshold),
                "track2p_pyrecest_calibrated_mht_candidates": int(
                    len(candidate_rows)
                ),
                "track2p_pyrecest_calibrated_mht_hypotheses": int(len(hypotheses)),
                "track2p_pyrecest_calibrated_mht_selected": int(len(selected_rows)),
                "track2p_pyrecest_calibrated_mht_applied": int(len(applied_keys)),
                "track2p_pyrecest_calibrated_mht_selected_score": float(
                    selected_hypothesis.score
                ),
                "track2p_pyrecest_calibrated_mht_feature_names": ",".join(
                    CALIBRATED_FEATURE_NAMES
                ),
            }
        )
        results.append(
            SubjectBenchmarkResult(
                subject=held_out.subject,
                variant="CoherenceSuffixStitch + calibrated PyRecEst residual MHT",
                method=cast(Any, METHOD),
                scores=scores,
                n_sessions=held_out.n_sessions,
                reference_source=GROUND_TRUTH_REFERENCE_SOURCE,
            )
        )

        ledger_rows.extend(
            _diagnostic_rows(
                held_out,
                gate=structural_gate,
                calibrator=calibrator,
                threshold=threshold,
                candidate_rows=candidate_rows,
                selected_rows=selected_rows,
                applied_keys=applied_keys,
                hypotheses=hypotheses,
                selected_hypothesis=selected_hypothesis,
                training_rows=training_structural_rows,
            )
        )

    return CalibratedResidualMHTResult(
        tuple(results),
        tuple(ledger_rows),
        tuple(residual_mht._summary_rows(ledger_rows)),
    )


def _edge_rows_for_state(
    state: Any,
    *,
    global_baseline_scores: Mapping[str, float],
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cell_probability_threshold: float,
    transform_type: str,
    structural_gate: cleanup.GrowthVetoGate,
) -> _SubjectRows:
    edge_rows = veto._accepted_edge_rows(
        state,
        global_baseline_scores=global_baseline_scores,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(cell_probability_threshold),
        transform_type=transform_type,
    )
    edge_rows = cleanup._augment_growth_veto_candidate_shifted_iou(
        edge_rows,
        state.sessions,
        gate=structural_gate,
        n_sessions=int(state.reference.shape[1]),
    )
    return _SubjectRows(
        subject=str(state.subject),
        n_sessions=int(state.reference.shape[1]),
        state=state,
        edge_rows=tuple(edge_rows),
    )


def _structural_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    gate: cleanup.GrowthVetoGate,
    n_sessions: int,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if _structural_candidate_gate_reason(row, gate=gate, n_sessions=n_sessions)
        == "accepted"
    ]


def _structural_candidate_gate_reason(
    row: Mapping[str, Any],
    *,
    gate: cleanup.GrowthVetoGate,
    n_sessions: int,
) -> str:
    """Return accepted using structural guards only, not residual thresholds."""

    if gate.require_not_suffix_edge and str(row.get("edge_source", "")) == "suffix":
        return "coherence_suffix_edge"
    if str(row.get("remove_reason", "")) != "split_edge":
        return "not_splittable"
    if int(row.get("would_split_component", 0)) <= 0:
        return "does_not_split_component"
    if gate.require_terminal_edge and int(row.get("is_terminal_edge", 0)) <= 0:
        return "not_terminal_edge"
    if gate.require_last_session_edge and int(row.get("is_last_session_edge", 0)) <= 0:
        return "not_last_session_edge"
    if gate.require_complete_component and int(
        row.get("complete_component_size", 0)
    ) < int(n_sessions):
        return "not_complete_component"
    if gate.min_complete_component_size is not None and int(
        row.get("complete_component_size", 0)
    ) < int(gate.min_complete_component_size):
        return "complete_component_size_below_gate"
    if int(row.get("growth_anchor_count", 0)) < max(0, int(gate.min_anchor_count)):
        return "growth_anchor_count_below_gate"

    row_rank = int(residual_mht._finite_float(row.get("row_rank"), float("inf")))
    column_rank = int(
        residual_mht._finite_float(row.get("column_rank"), float("inf"))
    )
    if row_rank <= 0 or row_rank > int(gate.max_row_rank):
        return "row_rank_above_gate"
    if column_rank <= 0 or column_rank > int(gate.max_column_rank):
        return "column_rank_above_gate"

    cell_a = residual_mht._finite_float(row.get("cell_probability_a"), float("nan"))
    cell_b = residual_mht._finite_float(row.get("cell_probability_b"), float("nan"))
    if not np.isfinite(cell_a) or not np.isfinite(cell_b):
        return "cell_probability_missing"
    min_cell_probability = min(cell_a, cell_b)
    if min_cell_probability < float(gate.min_cell_probability):
        return "cell_probability_below_gate"
    if gate.max_min_cell_probability is not None and min_cell_probability > float(
        gate.max_min_cell_probability
    ):
        return "min_cell_probability_above_gate"

    if any(not np.isfinite(value) for value in _calibrated_feature_vector(row)):
        return "calibrated_feature_missing"
    if gate.max_local_neighbor_distortion is not None:
        distortion = residual_mht._finite_float(
            row.get("local_neighbor_distortion"),
            float("nan"),
        )
        if not np.isfinite(distortion) or distortion > float(
            gate.max_local_neighbor_distortion
        ):
            return "local_neighbor_distortion_above_gate"
    return "accepted"


def _calibrated_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    gate: cleanup.GrowthVetoGate,
    n_sessions: int,
    calibrator: FalsePositiveCalibrator,
    threshold: float,
) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    for row in _structural_candidate_rows(rows, gate=gate, n_sessions=n_sessions):
        probability = calibrator.predict_false_positive_probability(row)
        score = _calibrated_log_likelihood_score(probability, threshold)
        if probability < threshold:
            continue
        candidates.append(
            {
                **dict(row),
                "pyrecest_candidate_id": residual_mht._candidate_id(row),
                "pyrecest_candidate_family": "calibrated_fp_probability",
                "pyrecest_candidate_score": float(score),
                "calibrated_fp_probability": float(probability),
                "calibrated_fp_threshold": float(threshold),
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["pyrecest_candidate_score"]),
            str(row["pyrecest_candidate_id"]),
        )
    )
    return candidates


def _to_calibrated_pyrecest_candidate(
    row: Mapping[str, Any],
) -> residual_mht.ResidualEditCandidate:
    return residual_mht.ResidualEditCandidate(
        candidate_id=str(row["pyrecest_candidate_id"]),
        score=float(row["pyrecest_candidate_score"]),
        conflict_keys=residual_mht._conflict_keys(row),
        metadata={
            "subject": str(row.get("subject", "")),
            "session_a": int(row.get("session_a", -1)),
            "session_b": int(row.get("session_b", -1)),
            "roi_a": int(row.get("roi_a", -1)),
            "roi_b": int(row.get("roi_b", -1)),
            "calibrated_fp_probability": float(
                row.get("calibrated_fp_probability", float("nan"))
            ),
        },
    )


def _fit_false_positive_calibrator(
    rows: Sequence[Mapping[str, Any]],
    *,
    options: CalibratedResidualMHTOptions,
) -> FalsePositiveCalibrator:
    labels = np.asarray([_false_positive_label(row) for row in rows], dtype=int)
    if (
        len(rows) == 0
        or int(np.sum(labels)) < int(options.min_training_positive_examples)
        or len(set(labels.tolist())) < 2
    ):
        return _constant_false_positive_calibrator(0.0)

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - sklearn is a project dependency
        raise ImportError(
            "Calibrated residual MHT requires scikit-learn for logistic calibration."
        ) from exc

    features = np.asarray([_calibrated_feature_vector(row) for row in rows], dtype=float)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    normalized = (features - mean) / scale
    estimator = LogisticRegression(
        class_weight="balanced",
        C=float(options.logistic_c),
        random_state=0,
        max_iter=10000,
    )
    estimator.fit(normalized, labels)
    return FalsePositiveCalibrator(
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
    )


def _constant_false_positive_calibrator(probability: float) -> FalsePositiveCalibrator:
    probability = _clamped_probability(probability)
    return FalsePositiveCalibrator(
        feature_mean=(0.0,) * len(CALIBRATED_FEATURE_NAMES),
        feature_scale=(1.0,) * len(CALIBRATED_FEATURE_NAMES),
        coefficients=(0.0,) * len(CALIBRATED_FEATURE_NAMES),
        intercept=_logit(probability),
    )


def _select_training_probability_threshold(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
) -> float:
    """Choose one fold-internal threshold using training labels/deltas only."""

    best: tuple[float, float, int, float] | None = None
    best_threshold = 1.0
    for threshold in sorted({float(value) for value in probabilities}, reverse=True):
        selected = [
            row for row, probability in zip(rows, probabilities, strict=True)
            if float(probability) >= threshold
        ]
        if _selected_has_training_tp_loss(selected):
            continue
        pairwise_fp_removed = -sum(
            residual_mht._finite_float(row.get("pairwise_fp_delta_if_removed"), 0.0)
            for row in selected
        )
        complete_fp_removed = -sum(
            residual_mht._finite_float(row.get("complete_fp_delta_if_removed"), 0.0)
            for row in selected
        )
        candidate_score = (
            float(complete_fp_removed),
            float(pairwise_fp_removed),
            -len(selected),
            -float(threshold),
        )
        if best is None or candidate_score > best:
            best = candidate_score
            best_threshold = float(threshold)
    if best is None:
        return float(math.nextafter(1.0, math.inf))
    return float(best_threshold)


def _selected_has_training_tp_loss(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        for key in ("pairwise_tp_delta_if_removed", "complete_tp_delta_if_removed"):
            if residual_mht._finite_float(row.get(key), 0.0) < 0.0:
                return True
        for key in ("pairwise_fn_delta_if_removed", "complete_fn_delta_if_removed"):
            if residual_mht._finite_float(row.get(key), 0.0) > 0.0:
                return True
    return False


def _false_positive_label(row: Mapping[str, Any]) -> int:
    return int(str(row.get("edge_status_against_gt", "")) == "false_positive")


def _calibrated_feature_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    raw_growth_residual = residual_mht._finite_float(
        row.get("growth_residual"),
        float("nan"),
    )
    raw_growth_mahalanobis = residual_mht._finite_float(
        row.get("growth_residual_mahalanobis"),
        float("nan"),
    )
    growth_residual = (
        max(0.0, raw_growth_residual)
        if np.isfinite(raw_growth_residual)
        else float("nan")
    )
    growth_mahalanobis = (
        max(0.0, raw_growth_mahalanobis)
        if np.isfinite(raw_growth_mahalanobis)
        else float("nan")
    )
    cell_a = residual_mht._finite_float(row.get("cell_probability_a"), float("nan"))
    cell_b = residual_mht._finite_float(row.get("cell_probability_b"), float("nan"))
    return (
        residual_mht._finite_float(row.get("registered_iou"), float("nan")),
        float(np.log1p(growth_residual)),
        float(np.log1p(growth_mahalanobis)),
        min(cell_a, cell_b),
    )


def _calibrated_log_likelihood_score(probability: float, threshold: float) -> float:
    return float(_logit(probability) - _logit(threshold))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return float(1.0 / (1.0 + z))
    z = math.exp(value)
    return float(z / (1.0 + z))


def _logit(probability: float) -> float:
    probability = _clamped_probability(probability)
    return float(math.log(probability / (1.0 - probability)))


def _clamped_probability(probability: float) -> float:
    return min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)


def _diagnostic_rows(
    held_out: _SubjectRows,
    *,
    gate: cleanup.GrowthVetoGate,
    calibrator: FalsePositiveCalibrator,
    threshold: float,
    candidate_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    applied_keys: Sequence[tuple[str, int, int, int, int]],
    hypotheses: Sequence[Any],
    selected_hypothesis: Any,
    training_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_keys = {cleanup._edge_row_key(row) for row in selected_rows}
    applied_set = set(applied_keys)
    candidates_by_id = {
        str(row["pyrecest_candidate_id"]): row for row in candidate_rows
    }
    hypothesis_ids = [";".join(h.candidate_ids) for h in hypotheses]
    output: list[dict[str, Any]] = []
    for row in held_out.edge_rows:
        key = cleanup._edge_row_key(row)
        candidate_id = residual_mht._candidate_id(row)
        structural_reason = _structural_candidate_gate_reason(
            row,
            gate=gate,
            n_sessions=held_out.n_sessions,
        )
        candidate_row = candidates_by_id.get(candidate_id)
        probability = (
            calibrator.predict_false_positive_probability(row)
            if structural_reason == "accepted"
            else float("nan")
        )
        score = (
            _calibrated_log_likelihood_score(probability, threshold)
            if np.isfinite(probability)
            else float("nan")
        )
        gate_reason = structural_reason
        if structural_reason == "accepted" and candidate_row is None:
            gate_reason = "calibrated_probability_below_threshold"
        output.append(
            {
                **row,
                "pyrecest_candidate_id": candidate_id,
                "pyrecest_candidate": int(candidate_row is not None),
                "pyrecest_candidate_score": float(score),
                "pyrecest_candidate_family": (
                    str(candidate_row.get("pyrecest_candidate_family", ""))
                    if candidate_row is not None
                    else ""
                ),
                "selected_by_pyrecest_mht": int(key in selected_keys),
                "applied_by_pyrecest_mht": int(key in applied_set),
                "pyrecest_selected_hypothesis": ";".join(
                    selected_hypothesis.candidate_ids
                ),
                "pyrecest_selected_hypothesis_score": float(selected_hypothesis.score),
                "pyrecest_hypothesis_count": int(len(hypotheses)),
                "pyrecest_top_hypotheses": "|".join(hypothesis_ids[:5]),
                "pyrecest_mht_gate_reason": gate_reason,
                "calibrated_fp_probability": float(probability),
                "calibrated_fp_threshold": float(threshold),
                "calibrated_fp_training_examples": int(len(training_rows)),
                "calibrated_fp_training_positive_examples": int(
                    sum(_false_positive_label(train_row) for train_row in training_rows)
                ),
            }
        )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = cleanup.build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-policy-pyrecest-calibrated-mht-cleanup"
    parser.description = (
        "Run LOSO-calibrated PyRecEst residual MHT over structurally valid "
        "CoherenceSuffixStitch edit candidates."
    )
    parser.set_defaults(growth_veto_base="coherence-suffix")
    parser.add_argument("--mht-max-edits-per-subject", type=int, default=4)
    parser.add_argument("--mht-max-hypotheses", type=int, default=64)
    parser.add_argument("--mht-edit-penalty", type=float, default=0.0)
    parser.add_argument("--mht-score-threshold", type=float, default=0.0)
    parser.add_argument("--calibrated-fp-logistic-c", type=float, default=0.5)
    parser.add_argument("--calibrated-fp-min-training-positives", type=int, default=1)
    return parser


def _explicit_option_present(args: Sequence[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in args)


def _reject_unsupported_calibrated_options(
    parser: argparse.ArgumentParser, args: Sequence[str]
) -> None:
    unsupported = sorted(
        option
        for option in _UNSUPPORTED_CALIBRATED_GROWTH_VETO_OPTIONS
        if _explicit_option_present(args, option)
    )
    if unsupported:
        parser.error(
            "track2p-policy-pyrecest-calibrated-mht-cleanup does not apply "
            "growth-residual, overlap, shifted-IoU, or deterministic veto-count "
            "gates; use MHT/calibration options instead of "
            + ", ".join(unsupported)
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    _reject_unsupported_calibrated_options(parser, raw_args)
    args = parser.parse_args(raw_args)
    if args.growth_veto_base != "coherence-suffix":
        parser.error(
            "track2p-policy-pyrecest-calibrated-mht-cleanup requires "
            "--growth-veto-base coherence-suffix"
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
        max_gap=args.max_gap,
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
    result = run_track2p_policy_pyrecest_calibrated_mht_cleanup(
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
        structural_gate=cleanup.GrowthVetoGate(
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
            max_vetoes_per_subject=int(args.mht_max_edits_per_subject),
        ),
        mht_options=CalibratedResidualMHTOptions(
            max_edits_per_subject=int(args.mht_max_edits_per_subject),
            max_hypotheses=int(args.mht_max_hypotheses),
            edit_penalty=float(args.mht_edit_penalty),
            score_threshold=float(args.mht_score_threshold),
            logistic_c=float(args.calibrated_fp_logistic_c),
            min_training_positive_examples=max(
                0,
                int(args.calibrated_fp_min_training_positives),
            ),
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
    if args.summary_output is not None:
        veto.write_rows(
            result.summary_rows,
            args.summary_output,
            output_format=cast(Literal["csv", "json"], args.format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
