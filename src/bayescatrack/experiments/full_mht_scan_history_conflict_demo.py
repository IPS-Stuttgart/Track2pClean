"""Constructed conflict for scan-history FullMHT pruning.

This demo isolates the new scan-time history-dynamics layer from benchmark
scoring.  Two candidate identity histories are available after the same scans:

* a locally higher raw-score continuation whose second edge is a strong
  within-history motion outlier;
* a slightly lower raw-score continuation whose edge diagnostics remain coherent.

Plain local-score pruning keeps the first candidate.  Scan-history-aware pruning
keeps the coherent candidate once the label-free motion-history risk is applied.
No references, benchmark scores, or audit labels are read by this module.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayescatrack.experiments.full_mht_scan_history_dynamics_integration import (
    scan_motion_history_risk,
)

METHOD = "full-mht-scan-history-conflict-demo"


@dataclass(frozen=True)
class ScanHistoryConflictCandidate:
    """One complete candidate identity history for the constructed conflict."""

    name: str
    description: str
    tracks: np.ndarray
    raw_score: float
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ScanHistoryConflictResult:
    """Selected history under one pruning policy."""

    policy: str
    description: str
    selected_candidate: str
    path: tuple[int, ...]
    raw_score: float
    history_risk: float
    pruning_score: float
    scan_motion_history_weight: float


def build_scan_history_conflict_candidates() -> list[ScanHistoryConflictCandidate]:
    """Return candidates where local score and history coherence disagree."""

    shared_first_edge = _edge_summary(
        0,
        1,
        1,
        20,
        registered_iou=0.90,
        shifted_iou=0.90,
        growth_residual=1.00,
        growth_mahalanobis=1.00,
        local_deformation=0.05,
    )
    coherent_second_edge = _edge_summary(
        1,
        2,
        20,
        30,
        registered_iou=0.86,
        shifted_iou=0.86,
        growth_residual=1.20,
        growth_mahalanobis=1.20,
        local_deformation=0.06,
    )
    motion_break_second_edge = _edge_summary(
        1,
        2,
        20,
        41,
        registered_iou=0.30,
        shifted_iou=0.25,
        growth_residual=6.50,
        growth_mahalanobis=7.00,
        local_deformation=1.00,
    )
    return [
        ScanHistoryConflictCandidate(
            name="locally_high_score_motion_break",
            description="higher raw scan score, but the second edge breaks the identity motion history",
            tracks=np.asarray([[1, 20, 41]], dtype=int),
            raw_score=10.0,
            history=(
                {"selected_edge_summaries": shared_first_edge},
                {"selected_edge_summaries": motion_break_second_edge},
            ),
        ),
        ScanHistoryConflictCandidate(
            name="lower_score_motion_coherent",
            description="lower raw scan score, but the full identity history stays coherent",
            tracks=np.asarray([[1, 20, 30]], dtype=int),
            raw_score=9.2,
            history=(
                {"selected_edge_summaries": shared_first_edge},
                {"selected_edge_summaries": coherent_second_edge},
            ),
        ),
    ]


def evaluate_scan_history_conflict_demo(
    *, scan_motion_history_weight: float = 1.0
) -> list[ScanHistoryConflictResult]:
    """Compare local-score pruning with scan-history-aware pruning."""

    candidates = build_scan_history_conflict_candidates()
    local_choice = max(candidates, key=lambda candidate: float(candidate.raw_score))
    history_choice = max(
        candidates,
        key=lambda candidate: _pruning_score(
            candidate,
            scan_motion_history_weight=float(scan_motion_history_weight),
        ),
    )
    return [
        _result(
            "local_score_only",
            "prune by raw scan-assignment score only",
            local_choice,
            scan_motion_history_weight=float(scan_motion_history_weight),
        ),
        _result(
            "scan_history_pruning",
            "prune by raw score minus label-free motion-history risk",
            history_choice,
            scan_motion_history_weight=float(scan_motion_history_weight),
        ),
    ]


def candidate_rows(
    *, scan_motion_history_weight: float = 1.0
) -> list[dict[str, Any]]:
    """Return all candidate scores for audit-style reporting."""

    return [
        {
            "candidate": candidate.name,
            "description": candidate.description,
            "path": _path_string(candidate.tracks),
            "raw_score": float(candidate.raw_score),
            "history_risk": float(_history_risk(candidate)),
            "pruning_score": float(
                _pruning_score(
                    candidate,
                    scan_motion_history_weight=float(scan_motion_history_weight),
                )
            ),
            "scan_motion_history_weight": float(scan_motion_history_weight),
        }
        for candidate in build_scan_history_conflict_candidates()
    ]


def format_markdown(results: Sequence[ScanHistoryConflictResult]) -> str:
    header = (
        "| policy | selected candidate | path | raw score | history risk | pruning score |\n"
        "| --- | --- | --- | ---: | ---: | ---: |"
    )
    lines = [header]
    for result in results:
        lines.append(
            "| {policy} | {candidate} | {path} | {raw:.3f} | {risk:.3f} | {score:.3f} |".format(
                policy=result.policy,
                candidate=result.selected_candidate,
                path="->".join(str(value) for value in result.path),
                raw=result.raw_score,
                risk=result.history_risk,
                score=result.pruning_score,
            )
        )
    return "\n".join(lines)


def _result(
    policy: str,
    description: str,
    candidate: ScanHistoryConflictCandidate,
    *,
    scan_motion_history_weight: float,
) -> ScanHistoryConflictResult:
    return ScanHistoryConflictResult(
        policy=policy,
        description=description,
        selected_candidate=candidate.name,
        path=tuple(int(value) for value in np.asarray(candidate.tracks, dtype=int)[0]),
        raw_score=float(candidate.raw_score),
        history_risk=float(_history_risk(candidate)),
        pruning_score=float(
            _pruning_score(
                candidate,
                scan_motion_history_weight=float(scan_motion_history_weight),
            )
        ),
        scan_motion_history_weight=float(scan_motion_history_weight),
    )


def _history_risk(candidate: ScanHistoryConflictCandidate) -> float:
    hypothesis = type(
        "ScanHistoryConflictHypothesis",
        (),
        {"tracks": candidate.tracks, "history": candidate.history},
    )()
    return float(scan_motion_history_risk(hypothesis))


def _pruning_score(
    candidate: ScanHistoryConflictCandidate, *, scan_motion_history_weight: float
) -> float:
    return float(candidate.raw_score) - float(scan_motion_history_weight) * _history_risk(candidate)


def _edge_summary(
    session_a: int,
    session_b: int,
    roi_a: int,
    roi_b: int,
    *,
    registered_iou: float,
    shifted_iou: float,
    growth_residual: float,
    growth_mahalanobis: float,
    local_deformation: float,
) -> str:
    return (
        f"{int(session_a)}:{int(roi_a)}->{int(session_b)}:{int(roi_b)}"
        f"|reg={float(registered_iou):.3f}"
        f"|shift={float(shifted_iou):.3f}"
        f"|growth={float(growth_residual):.3f}"
        f"|mahal={float(growth_mahalanobis):.3f}"
        f"|local={float(local_deformation):.3f}"
    )


def _path_string(tracks: np.ndarray) -> str:
    row = np.asarray(tracks, dtype=int)[0]
    return "->".join(str(int(value)) for value in row)


def _result_rows(results: Sequence[ScanHistoryConflictResult]) -> list[dict[str, Any]]:
    return [
        {
            "policy": result.policy,
            "description": result.description,
            "selected_candidate": result.selected_candidate,
            "path": "->".join(str(value) for value in result.path),
            "raw_score": result.raw_score,
            "history_risk": result.history_risk,
            "pruning_score": result.pruning_score,
            "scan_motion_history_weight": result.scan_motion_history_weight,
        }
        for result in results
    ]


def _write_csv(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_scan_history_conflict_demo",
        description="Show that scan-history FullMHT pruning can reject a locally stronger but history-incoherent continuation.",
    )
    parser.add_argument("--scan-motion-history-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for selected-policy rows.")
    parser.add_argument("--candidate-output", type=Path, default=None, help="Optional CSV path for all candidate scores.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    weight = float(args.scan_motion_history_weight)
    results = evaluate_scan_history_conflict_demo(scan_motion_history_weight=weight)
    print(format_markdown(results))
    if args.output is not None:
        _write_csv(_result_rows(results), args.output)
    if args.candidate_output is not None:
        _write_csv(candidate_rows(scan_motion_history_weight=weight), args.candidate_output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
