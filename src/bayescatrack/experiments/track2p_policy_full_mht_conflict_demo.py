"""Controlled full-scan MHT demos where history beats greedy assignment.

The real FullMHT benchmark now opens a beam of complete track-table histories.
These synthetic demos isolate why that matters.  A locally stronger first-scan
edge can lead to a dead end at the next scan; a slightly weaker first edge can
admit a strong continuation and preserve the complete identity.  A greedy beam
of width one takes the local edge and loses the track.  A bounded MHT beam keeps
both first-scan hypotheses long enough for the globally better history to win.

The second demo embeds the same conflict among many stable tracks, so the greedy
solution remains pairwise-good while still damaging complete-track identity.  No
ground-truth labels enter the selection scores.  The reference matrix is used
only to report the same pairwise and complete-track metrics used by the Track2p
benchmark rows.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bayescatrack.evaluation.complete_track_scores import score_track_matrices

try:
    from pyrecest.utils import murty_k_best_assignments
except ImportError as exc:  # pragma: no cover - stale PyRecEst environment
    raise ImportError(
        "track2p-policy-full-mht-conflict-demo requires PyRecEst with "
        "pyrecest.utils.murty_k_best_assignments."
    ) from exc


METHOD = "track2p-policy-full-mht-conflict-demo"


@dataclass(frozen=True)
class DemoScenario:
    """A scan-assignment conflict where local and global optima differ."""

    name: str
    reference: np.ndarray
    seed_track: np.ndarray
    edge_scores: Mapping[tuple[int, int, int, int], float]


@dataclass(frozen=True)
class _DemoHypothesis:
    tracks: np.ndarray
    score: float
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DemoArmResult:
    arm: str
    description: str
    beam_width: int
    scan_hypotheses: int
    score: float
    path: tuple[int, ...]
    pairwise_f1: float
    complete_track_f1: float


def build_default_scenario() -> DemoScenario:
    """Return the canonical locally-good / globally-bad assignment scenario."""

    return DemoScenario(
        name="local-edge-dead-end",
        reference=np.asarray([[1, 20, 30]], dtype=int),
        seed_track=np.asarray([[1, -1, -1]], dtype=int),
        edge_scores={
            # Locally strongest edge.  Greedy takes this, then has no useful
            # continuation and pays the miss cost on the final scan.
            (0, 1, 1, 10): 5.0,
            # Slightly weaker first edge.  The MHT beam keeps it alive.
            (0, 1, 1, 20): 4.0,
            # Strong continuation available only after choosing ROI 20.
            (1, 2, 20, 30): 5.0,
        },
    )


def build_pairwise_good_complete_bad_scenario(
    *, stable_tracks: int = 20
) -> DemoScenario:
    """Embed one ambiguous identity in many easy tracks.

    This is the paper-facing toy case: local greedy assignment keeps a high
    pairwise score because most tracks are easy, but one locally attractive edge
    still breaks a complete identity.  The MHT beam can keep the weaker first
    edge alive until the later continuation makes the complete history win.
    """

    n_stable = max(1, int(stable_tracks))
    reference = np.full((n_stable + 1, 3), -1, dtype=int)
    seed_track = np.full_like(reference, -1)
    edge_scores: dict[tuple[int, int, int, int], float] = {
        (0, 1, 1, 10): 5.0,
        (0, 1, 1, 20): 4.0,
        (1, 2, 20, 30): 5.0,
    }
    reference[0] = np.asarray([1, 20, 30], dtype=int)
    seed_track[0, 0] = 1

    for index in range(n_stable):
        row = index + 1
        seed_roi = 1000 + index
        middle_roi = 2000 + index
        final_roi = 3000 + index
        reference[row] = np.asarray([seed_roi, middle_roi, final_roi], dtype=int)
        seed_track[row, 0] = seed_roi
        edge_scores[(0, 1, seed_roi, middle_roi)] = 6.0
        edge_scores[(1, 2, middle_roi, final_roi)] = 6.0

    return DemoScenario(
        name="pairwise-good-complete-bad",
        reference=reference,
        seed_track=seed_track,
        edge_scores=edge_scores,
    )


def evaluate_demo(
    *,
    scenario: DemoScenario | None = None,
    miss_cost: float = 2.0,
    mht_beam_width: int = 2,
    scan_hypotheses: int = 2,
) -> list[DemoArmResult]:
    """Compare greedy beam-1 assignment against full-history MHT."""

    scenario = scenario or build_default_scenario()
    greedy = _run_assignment_beam(
        scenario,
        beam_width=1,
        scan_hypotheses=max(1, int(scan_hypotheses)),
        miss_cost=float(miss_cost),
    )
    mht = _run_assignment_beam(
        scenario,
        beam_width=max(1, int(mht_beam_width)),
        scan_hypotheses=max(1, int(scan_hypotheses)),
        miss_cost=float(miss_cost),
    )
    return [
        _arm_result(
            "greedy",
            "beam width 1; keeps only the locally best scan edge",
            greedy,
            reference=scenario.reference,
            beam_width=1,
            scan_hypotheses=max(1, int(scan_hypotheses)),
        ),
        _arm_result(
            "full_mht",
            "beam keeps alternate identity history until later evidence arrives",
            mht,
            reference=scenario.reference,
            beam_width=max(1, int(mht_beam_width)),
            scan_hypotheses=max(1, int(scan_hypotheses)),
        ),
    ]


def _run_assignment_beam(
    scenario: DemoScenario,
    *,
    beam_width: int,
    scan_hypotheses: int,
    miss_cost: float,
) -> _DemoHypothesis:
    hypotheses = [_DemoHypothesis(scenario.seed_track.copy(), 0.0, tuple())]
    n_sessions = int(scenario.seed_track.shape[1])
    for session_index in range(n_sessions - 1):
        expanded: list[_DemoHypothesis] = []
        for hypothesis in hypotheses:
            expanded.extend(
                _advance_demo_scan(
                    hypothesis,
                    edge_scores=scenario.edge_scores,
                    session_index=session_index,
                    scan_hypotheses=scan_hypotheses,
                    miss_cost=float(miss_cost),
                )
            )
        expanded.sort(key=lambda item: -float(item.score))
        hypotheses = expanded[: max(1, int(beam_width))]
    return hypotheses[0]


def _advance_demo_scan(
    hypothesis: _DemoHypothesis,
    *,
    edge_scores: Mapping[tuple[int, int, int, int], float],
    session_index: int,
    scan_hypotheses: int,
    miss_cost: float,
) -> list[_DemoHypothesis]:
    tracks = np.asarray(hypothesis.tracks, dtype=int)
    next_session = int(session_index) + 1
    active_rows = [
        row_index
        for row_index, row in enumerate(tracks)
        if int(row[int(session_index)]) >= 0
    ]
    if not active_rows:
        return [_append_demo_history(hypothesis, session_index, 0.0, 0, 0)]
    target_rois = sorted(
        {
            int(roi_b)
            for (session_a, session_b, roi_a, roi_b), _score in edge_scores.items()
            if int(session_a) == int(session_index)
            and int(session_b) == next_session
            and any(int(tracks[row, session_index]) == int(roi_a) for row in active_rows)
        }
    )
    if not target_rois:
        updated = tracks.copy()
        updated[active_rows, next_session] = -1
        return [
            _append_demo_history(
                _DemoHypothesis(
                    updated, hypothesis.score - float(miss_cost), hypothesis.history
                ),
                session_index,
                float(miss_cost),
                0,
                len(active_rows),
            )
        ]
    cost_matrix = np.full((len(active_rows), len(target_rois)), np.inf, dtype=float)
    for row_pos, row_index in enumerate(active_rows):
        roi_a = int(tracks[int(row_index), int(session_index)])
        for col, roi_b in enumerate(target_rois):
            score = edge_scores.get((int(session_index), next_session, roi_a, int(roi_b)))
            if score is not None:
                cost_matrix[int(row_pos), int(col)] = -float(score)
    solutions = murty_k_best_assignments(
        cost_matrix,
        k=max(1, int(scan_hypotheses)),
        row_non_assignment_costs=np.full((len(active_rows),), float(miss_cost)),
        col_non_assignment_costs=np.zeros((len(target_rois),), dtype=float),
    )
    output: list[_DemoHypothesis] = []
    for solution in solutions:
        assignment = np.asarray(solution["assignment"], dtype=int)
        updated = tracks.copy()
        assigned = 0
        missed = 0
        for row_pos, row_index in enumerate(active_rows):
            col = int(assignment[int(row_pos)])
            if col >= 0:
                updated[int(row_index), next_session] = int(target_rois[col])
                assigned += 1
            else:
                updated[int(row_index), next_session] = -1
                missed += 1
        output.append(
            _append_demo_history(
                _DemoHypothesis(
                    updated,
                    float(hypothesis.score) - float(solution["cost"]),
                    hypothesis.history,
                ),
                session_index,
                float(solution["cost"]),
                assigned,
                missed,
            )
        )
    return output


def _append_demo_history(
    hypothesis: _DemoHypothesis,
    session_index: int,
    scan_cost: float,
    assigned: int,
    missed: int,
) -> _DemoHypothesis:
    return _DemoHypothesis(
        hypothesis.tracks,
        float(hypothesis.score),
        hypothesis.history
        + (
            {
                "session_index": int(session_index),
                "scan_cost": float(scan_cost),
                "assigned_edges": int(assigned),
                "missed_tracks": int(missed),
            },
        ),
    )


def _arm_result(
    arm: str,
    description: str,
    hypothesis: _DemoHypothesis,
    *,
    reference: np.ndarray,
    beam_width: int,
    scan_hypotheses: int,
) -> DemoArmResult:
    scores = score_track_matrices(hypothesis.tracks, np.asarray(reference, dtype=int))
    return DemoArmResult(
        arm=arm,
        description=description,
        beam_width=int(beam_width),
        scan_hypotheses=int(scan_hypotheses),
        score=float(hypothesis.score),
        path=tuple(int(value) for value in hypothesis.tracks[0]),
        pairwise_f1=float(scores["pairwise_f1"]),
        complete_track_f1=float(scores["complete_track_f1"]),
    )


def format_markdown(results: Sequence[DemoArmResult]) -> str:
    header = (
        "| arm | strategy | beam | score | path | pairwise F1 | complete-track F1 |\n"
        "| --- | --- | ---: | ---: | --- | ---: | ---: |"
    )
    lines = [header]
    for result in results:
        lines.append(
            "| {arm} | {description} | {beam_width} | {score:.3f} | {path} | "
            "{pairwise:.3f} | {complete:.3f} |".format(
                arm=result.arm,
                description=result.description,
                beam_width=result.beam_width,
                score=result.score,
                path="->".join(str(value) for value in result.path),
                pairwise=result.pairwise_f1,
                complete=result.complete_track_f1,
            )
        )
    return "\n".join(lines)


def _result_rows(results: Sequence[DemoArmResult]) -> list[dict[str, Any]]:
    return [
        {
            "arm": result.arm,
            "strategy": result.description,
            "beam_width": result.beam_width,
            "scan_hypotheses": result.scan_hypotheses,
            "score": result.score,
            "path": "->".join(str(value) for value in result.path),
            "pairwise_f1": result.pairwise_f1,
            "complete_track_f1": result.complete_track_f1,
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
        prog="bayescatrack benchmark track2p-policy-full-mht-conflict-demo",
        description=(
            "Demonstrate that a full scan-assignment MHT beam can choose a "
            "globally better identity history than greedy local assignment."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path.")
    parser.add_argument("--miss-cost", type=float, default=2.0)
    parser.add_argument("--mht-beam-width", type=int, default=2)
    parser.add_argument("--scan-hypotheses", type=int, default=2)
    parser.add_argument(
        "--scenario",
        choices=("local-edge-dead-end", "pairwise-good-complete-bad"),
        default="local-edge-dead-end",
    )
    parser.add_argument("--stable-tracks", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenario = (
        build_pairwise_good_complete_bad_scenario(stable_tracks=int(args.stable_tracks))
        if args.scenario == "pairwise-good-complete-bad"
        else build_default_scenario()
    )
    results = evaluate_demo(
        scenario=scenario,
        miss_cost=float(args.miss_cost),
        mht_beam_width=max(1, int(args.mht_beam_width)),
        scan_hypotheses=max(1, int(args.scan_hypotheses)),
    )
    print(format_markdown(results))
    if args.output is not None:
        _write_csv(_result_rows(results), args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
