"""Controlled demonstration that PyRecEst residual-MHT is not inert.

Across the real Track2p rows the bounded residual-MHT reduces to deterministic
gating, because the growth-veto / structural gate only admits edges that are the
unique best assignment in their row and column (``max_row_rank == 1`` and
``max_column_rank == 1``).  Competing edges -- the two halves of a Track2p
merge/collision, where one target ROI is claimed by two predecessors -- are
therefore never candidates, so the MHT has no conflict to arbitrate and selects
exactly the gated set.

This module isolates the regime where the framework *does* earn its place: a
merge collision is admitted as two competing removal candidates that share a
target ROI.  Resolving such a collision is a joint decision -- you should sever
the worse edge while keeping the better one, not both and not neither.  We
compare, on the same candidate set and the same label-free removal scores:

* ``deterministic`` -- the current rank-gated behaviour: collision edges are
  screened out (they have a competitor), so only the non-competing false edge is
  removed and the collision is left uncorrected;
* ``greedy`` -- relax the rank gate but keep per-edge, conflict-blind selection:
  every gated candidate is removed, which severs the *true* collision edge as
  well and destroys a complete track;
* ``mht`` -- relax the rank gate and let PyRecEst's bounded residual-MHT select a
  globally compatible edit set: it keeps the conflicting pair mutually exclusive,
  removes only the worse collision edge, and repairs the merge.

The selector, the conflict keys, the track-split, and the metric are the real
pipeline functions; only the candidate set is synthetic.  Selection never reads
any ground-truth field (the ``edge_truth`` column is reporting-only), mirroring
the no-leakage contract of the benchmark rows.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
from bayescatrack.experiments import track2p_policy_growth_veto_whatif as veto
from bayescatrack.experiments import (
    track2p_policy_pyrecest_residual_mht_cleanup as residual_mht,
)
from pyrecest.tracking import (
    ResidualEditCandidate,
    ResidualMHTConfig,
    select_residual_hypothesis,
)

METHOD = "track2p-policy-pyrecest-mht-conflict-demo"


def _positive_int_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
    else:
        raise ValueError(f"{name} must be a positive integer")
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_int_arg(value: str) -> int:
    try:
        return _positive_int_value(value, name="value")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


@dataclass(frozen=True)
class DemoScenario:
    """A synthetic prediction with a Track2p merge collision plus candidates."""

    name: str
    reference: np.ndarray
    predicted: np.ndarray
    candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ArmResult:
    """One selection strategy applied to a scenario."""

    arm: str
    description: str
    removed_candidate_ids: tuple[str, ...]
    n_edits: int
    conflict_free: bool
    pairwise_f1: float
    complete_track_f1: float


def _candidate_row(
    *,
    subject: str,
    session_a: int,
    roi_a: int,
    session_b: int,
    roi_b: int,
    removal_score: float,
    edge_truth: str,
) -> dict[str, Any]:
    """Build a candidate edge-removal row in the benchmark ledger schema."""

    return {
        "subject": subject,
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "occurrence_index": 0,
        # Label-free residual removal score; a stand-in for any score the cleanup
        # already computes (growth residual, registered/shifted IoU, ...).
        "removal_score": float(removal_score),
        # Reporting-only ground-truth tag. Never read by any selector below.
        "edge_truth": str(edge_truth),
    }


def build_default_scenario() -> DemoScenario:
    """Return the canonical merge-collision demonstration scenario.

    Three true complete tracks over three sessions.  Track2p produced two
    errors at the second session transition:

    * a *merge collision*: ROI ``12`` at session 2 is claimed by both ``11`` and
      ``21`` (its true predecessor ``21 -> 22`` is dropped);
    * an independent *false continuation*: ``31 -> 99`` (true successor ``32`` is
      dropped, ``99`` is spurious).
    """

    subject = "demo"
    reference = np.asarray(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ],
        dtype=int,
    )
    predicted = np.asarray(
        [
            [10, 11, 12],  # true track A (kept)
            [20, 21, 12],  # collision: 21 -> 12 is false (should be 21 -> 22)
            [30, 31, 99],  # false continuation: 31 -> 99 (should be 31 -> 32)
        ],
        dtype=int,
    )
    candidates = (
        # The true half of the collision: a low removal score, but it still
        # clears any gate threshold, so conflict-blind selection will take it.
        _candidate_row(
            subject=subject,
            session_a=1,
            roi_a=11,
            session_b=2,
            roi_b=12,
            removal_score=1.0,
            edge_truth="true_positive",
        ),
        # The false half of the collision: high removal score.
        _candidate_row(
            subject=subject,
            session_a=1,
            roi_a=21,
            session_b=2,
            roi_b=12,
            removal_score=2.6,
            edge_truth="false_positive",
        ),
        # The independent false continuation: high removal score, no competitor.
        _candidate_row(
            subject=subject,
            session_a=1,
            roi_a=31,
            session_b=2,
            roi_b=99,
            removal_score=2.2,
            edge_truth="false_positive",
        ),
    )
    return DemoScenario(
        name="merge-collision",
        reference=reference,
        predicted=predicted,
        candidates=candidates,
    )


def _predicted_edges(predicted: np.ndarray) -> list[tuple[int, int, int, int]]:
    matrix = np.asarray(predicted, dtype=int)
    edges: list[tuple[int, int, int, int]] = []
    for row in matrix:
        for session_index in range(matrix.shape[1] - 1):
            roi_a = int(row[session_index])
            roi_b = int(row[session_index + 1])
            if roi_a >= 0 and roi_b >= 0:
                edges.append((session_index, session_index + 1, roi_a, roi_b))
    return edges


def _has_competitor(predicted: np.ndarray, row: Mapping[str, Any]) -> bool:
    """Return whether the edge shares a source or target ROI with another edge.

    This is a transparent proxy for the gate's ``max_row_rank == 1`` and
    ``max_column_rank == 1`` rule: a contested source/target is a rank > 1
    assignment, which the deterministic cleanup refuses to edit.
    """

    edges = _predicted_edges(predicted)
    session_a = int(row["session_a"])
    session_b = int(row["session_b"])
    roi_a = int(row["roi_a"])
    roi_b = int(row["roi_b"])
    target_count = sum(
        1 for (_sa, sb, _ea, eb) in edges if sb == session_b and eb == roi_b
    )
    source_count = sum(
        1 for (sa, _sb, ea, _eb) in edges if sa == session_a and ea == roi_a
    )
    return target_count > 1 or source_count > 1


def deterministic_rank_gated_select(
    scenario: DemoScenario,
) -> list[dict[str, Any]]:
    """Select non-competing candidates only (current rank-gated behaviour)."""

    return [
        row
        for row in scenario.candidates
        if not _has_competitor(scenario.predicted, row)
    ]


def greedy_select(
    candidates: Sequence[Mapping[str, Any]],
    *,
    score_threshold: float,
    max_edits: int,
) -> list[Mapping[str, Any]]:
    """Conflict-blind per-edge selection: take every gated candidate by score."""

    max_edits = _positive_int_value(max_edits, name="max_edits")
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row["removal_score"]),
            residual_mht._candidate_id(row),
        ),
    )
    selected = [
        row for row in ordered if float(row["removal_score"]) >= float(score_threshold)
    ]
    return selected[:max_edits]


def mht_select(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: ResidualMHTConfig,
) -> list[Mapping[str, Any]]:
    """Conflict-aware joint selection via PyRecEst bounded residual-MHT."""

    pyrecest_candidates = [
        ResidualEditCandidate(
            candidate_id=residual_mht._candidate_id(row),
            score=float(row["removal_score"]),
            conflict_keys=residual_mht._conflict_keys(row),
        )
        for row in candidates
    ]
    chosen = select_residual_hypothesis(pyrecest_candidates, config=config)
    chosen_ids = set(chosen.candidate_ids)
    return [
        row
        for row in candidates
        if residual_mht._candidate_id(row) in chosen_ids
    ]


def apply_removals(
    predicted: np.ndarray, rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Apply edge removals with the real pipeline split (`_remove_edge_occurrence`)."""

    matrix = np.asarray(predicted, dtype=int)
    for row in rows:
        edge = (
            int(row["session_a"]),
            int(row["session_b"]),
            int(row["roi_a"]),
            int(row["roi_b"]),
        )
        split = veto._remove_edge_occurrence(
            matrix, edge, occurrence_index=int(row.get("occurrence_index", 0))
        )
        matrix = np.asarray(split.tracks, dtype=int)
    return matrix


def is_conflict_free(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether a removal set has pairwise-disjoint conflict keys."""

    seen: set[str] = set()
    for row in rows:
        keys = set(residual_mht._conflict_keys(row))
        if seen & keys:
            return False
        seen |= keys
    return True


def _arm_result(
    arm: str,
    description: str,
    scenario: DemoScenario,
    removed: Sequence[Mapping[str, Any]],
) -> ArmResult:
    tracks = apply_removals(scenario.predicted, removed)
    scores = score_track_matrices(tracks, scenario.reference)
    return ArmResult(
        arm=arm,
        description=description,
        removed_candidate_ids=tuple(
            residual_mht._candidate_id(row) for row in removed
        ),
        n_edits=len(removed),
        conflict_free=is_conflict_free(removed),
        pairwise_f1=float(scores["pairwise_f1"]),
        complete_track_f1=float(scores["complete_track_f1"]),
    )


def evaluate_scenario(
    scenario: DemoScenario,
    *,
    score_threshold: float = 0.0,
    edit_penalty: float = 0.25,
    max_edits: int = 3,
    max_hypotheses: int = 64,
) -> list[ArmResult]:
    """Return the baseline / deterministic / greedy / MHT comparison arms."""

    max_edits = _positive_int_value(max_edits, name="max_edits")
    config = ResidualMHTConfig(
        max_edits=max_edits,
        max_hypotheses=int(max_hypotheses),
        edit_penalty=float(edit_penalty),
        score_threshold=float(score_threshold),
        include_empty=True,
    )
    deterministic = deterministic_rank_gated_select(scenario)
    greedy = greedy_select(
        scenario.candidates,
        score_threshold=score_threshold,
        max_edits=max_edits,
    )
    mht = mht_select(scenario.candidates, config=config)
    return [
        _arm_result("baseline", "no cleanup", scenario, []),
        _arm_result(
            "deterministic",
            "rank-gated, conflict edges screened out (current pipeline)",
            scenario,
            deterministic,
        ),
        _arm_result(
            "greedy",
            "relaxed gate, conflict-blind per-edge removal",
            scenario,
            greedy,
        ),
        _arm_result(
            "mht",
            "relaxed gate, PyRecEst conflict-aware joint selection",
            scenario,
            mht,
        ),
    ]


def format_markdown(results: Sequence[ArmResult]) -> str:
    """Render the comparison arms as a Markdown table."""

    header = (
        "| arm | strategy | edits | conflict-free | pairwise F1 | complete-track F1 |\n"
        "| --- | --- | ---: | :---: | ---: | ---: |"
    )
    lines = [header]
    for result in results:
        lines.append(
            "| {arm} | {description} | {n_edits} | {conflict} | "
            "{pairwise:.4f} | {complete:.4f} |".format(
                arm=result.arm,
                description=result.description,
                n_edits=result.n_edits,
                conflict="yes" if result.conflict_free else "no",
                pairwise=result.pairwise_f1,
                complete=result.complete_track_f1,
            )
        )
    return "\n".join(lines)


def _results_as_rows(results: Sequence[ArmResult]) -> list[dict[str, Any]]:
    return [
        {
            "arm": result.arm,
            "strategy": result.description,
            "n_edits": result.n_edits,
            "removed_candidate_ids": ";".join(result.removed_candidate_ids),
            "conflict_free": int(result.conflict_free),
            "pairwise_f1": result.pairwise_f1,
            "complete_track_f1": result.complete_track_f1,
        }
        for result in results
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-pyrecest-mht-conflict-demo",
        description=(
            "Controlled demonstration that PyRecEst residual-MHT resolves a "
            "Track2p merge collision that conflict-blind gating corrupts."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path.")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--edit-penalty", type=float, default=0.25)
    parser.add_argument("--max-edits", type=_positive_int_arg, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenario = build_default_scenario()
    results = evaluate_scenario(
        scenario,
        score_threshold=float(args.score_threshold),
        edit_penalty=float(args.edit_penalty),
        max_edits=args.max_edits,
    )
    print(format_markdown(results))
    if args.output is not None:
        veto.write_rows(_results_as_rows(results), args.output, output_format="csv")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
