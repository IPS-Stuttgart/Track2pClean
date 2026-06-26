"""Opt-in scan-time motion-history pruning for FullMHT.

The terminal history-dynamics objective can rerank complete hypotheses at the end
of a subject.  This module moves the same label-free idea into beam pruning: at
each scan, hypotheses whose selected identity histories already contain a strong
within-history motion outlier lose pruning score.  That gives MHT a genuine
full-history selection pressure before locally plausible but globally bad
histories can crowd alternatives out of the beam.

The implementation deliberately uses only diagnostic features already produced by
FullMHT selected-edge summaries.  It does not load references, benchmark scores,
or audit labels.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_REGISTERED_IOU_DROP = 0.15
_SHIFTED_IOU_DROP = 0.20
_GROWTH_RESIDUAL_OFFSET = 2.50
_GROWTH_MAHALANOBIS_OFFSET = 2.00
_LOCAL_DEFORMATION_OFFSET = 0.35
_MISSING_FEATURE_PENALTY = 1.00
_EDGE_RE = re.compile(r"^(?P<sa>\d+):(?P<ra>-?\d+)->(?P<sb>\d+):(?P<rb>-?\d+)$")


@dataclass(frozen=True)
class ScanHistoryEdgeFeatures:
    """Label-free diagnostics for one selected edge in a scan-time history."""

    edge: tuple[int, int, int, int]
    registered_iou: float
    shifted_iou: float
    growth_residual: float
    growth_mahalanobis: float
    local_deformation: float


def install_full_mht_scan_history_dynamics_pruning() -> None:
    """Install scan-time motion-history risk into FullMHT beam pruning."""

    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    if getattr(full_mht, "_bayescatrack_scan_history_dynamics_pruning", False):
        return

    original_score = full_mht._beam_pruning_score

    def _beam_pruning_score_with_scan_history(hypothesis: Any, *, config: Any) -> float:
        score = float(original_score(hypothesis, config=config))
        weight = _scan_motion_history_weight(config)
        if weight <= 0.0:
            return score
        return float(score - weight * scan_motion_history_risk(hypothesis))

    full_mht._beam_pruning_score = _beam_pruning_score_with_scan_history  # type: ignore[method-assign]
    full_mht._bayescatrack_scan_history_dynamics_original_score = original_score
    full_mht._bayescatrack_scan_history_dynamics_pruning = True


def scan_motion_history_risk(hypothesis: Any) -> float:
    """Compute label-free motion-history risk from selected scan summaries."""

    tracks = np.asarray(getattr(hypothesis, "tracks", hypothesis), dtype=int)
    if tracks.ndim != 2 or tracks.size == 0:
        return 0.0
    feature_map = _selected_edge_feature_map(getattr(hypothesis, "history", ()))
    if not feature_map:
        return 0.0

    risk = 0.0
    for row in tracks:
        observed_sessions = np.flatnonzero(np.asarray(row, dtype=int) >= 0)
        if observed_sessions.size < 3:
            continue
        features: list[ScanHistoryEdgeFeatures] = []
        missing_features = 0
        for left, right in zip(observed_sessions[:-1], observed_sessions[1:]):
            edge = (
                int(left),
                int(right),
                int(row[int(left)]),
                int(row[int(right)]),
            )
            edge_features = feature_map.get(edge)
            if edge_features is None:
                missing_features += 1
            else:
                features.append(edge_features)
        risk += row_scan_motion_history_risk(
            features,
            missing_features=missing_features,
        )
    return float(risk)


def row_scan_motion_history_risk(
    features: Sequence[ScanHistoryEdgeFeatures], *, missing_features: int = 0
) -> float:
    """Return robust within-history risk for one partial identity history."""

    risk = max(0, int(missing_features)) * _MISSING_FEATURE_PENALTY
    if len(features) < 2:
        return float(risk)

    registered = _feature_array(features, "registered_iou")
    shifted = _feature_array(features, "shifted_iou")
    growth = _feature_array(features, "growth_residual")
    mahalanobis = _feature_array(features, "growth_mahalanobis")
    local = _feature_array(features, "local_deformation")

    risk += _low_outlier_risk(registered, allowed_drop=_REGISTERED_IOU_DROP)
    risk += _low_outlier_risk(shifted, allowed_drop=_SHIFTED_IOU_DROP)
    risk += _high_outlier_risk(growth, allowed_offset=_GROWTH_RESIDUAL_OFFSET)
    risk += _high_outlier_risk(mahalanobis, allowed_offset=_GROWTH_MAHALANOBIS_OFFSET)
    risk += _high_outlier_risk(local, allowed_offset=_LOCAL_DEFORMATION_OFFSET)
    return float(risk)


def parse_selected_edge_summary(summary: str) -> ScanHistoryEdgeFeatures | None:
    """Parse one FullMHT selected-edge summary into numeric diagnostics."""

    parts = [part for part in str(summary).split("|") if part]
    if not parts:
        return None
    match = _EDGE_RE.match(parts[0].strip())
    if match is None:
        return None
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    required = ("reg", "shift", "growth", "mahal", "local")
    if any(key not in values for key in required):
        return None
    try:
        return ScanHistoryEdgeFeatures(
            edge=(
                int(match.group("sa")),
                int(match.group("sb")),
                int(match.group("ra")),
                int(match.group("rb")),
            ),
            registered_iou=float(values["reg"]),
            shifted_iou=float(values["shift"]),
            growth_residual=float(values["growth"]),
            growth_mahalanobis=float(values["mahal"]),
            local_deformation=float(values["local"]),
        )
    except ValueError:
        return None


def _selected_edge_feature_map(
    history: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, int, int], ScanHistoryEdgeFeatures]:
    features: dict[tuple[int, int, int, int], ScanHistoryEdgeFeatures] = {}
    for scan in history:
        raw = str(scan.get("selected_edge_summaries", ""))
        if not raw:
            continue
        for item in raw.split(";"):
            parsed = parse_selected_edge_summary(item)
            if parsed is not None:
                features[parsed.edge] = parsed
    return features


def _feature_array(features: Sequence[ScanHistoryEdgeFeatures], name: str) -> np.ndarray:
    return np.asarray([float(getattr(feature, name)) for feature in features], dtype=float)


def _low_outlier_risk(values: np.ndarray, *, allowed_drop: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0
    reference = float(np.median(finite))
    return float(np.sum(np.maximum(0.0, reference - finite - float(allowed_drop))))


def _high_outlier_risk(values: np.ndarray, *, allowed_offset: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0
    reference = float(np.median(finite))
    return float(np.sum(np.maximum(0.0, finite - reference - float(allowed_offset))))


def _scan_motion_history_weight(config: Any) -> float:
    try:
        return max(0.0, float(getattr(config, "scan_motion_history_weight", 0.0)))
    except (TypeError, ValueError):
        return 0.0
