from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import cli
from bayescatrack.experiments import (
    track2p_policy_coherence_teacher_overlay_audit as overlay_audit,
)


def test_coherence_teacher_overlay_audit_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-coherence-teacher-overlay-audit"]

    assert canonical == "track2p-policy-coherence-teacher-overlay-audit"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-coherence-teacher-overlay-audit"]
        == canonical
    )
    assert (
        cli._BENCHMARK_COMMANDS[canonical].module
        == "bayescatrack.experiments.track2p_policy_coherence_teacher_overlay_audit"
    )


def test_coherence_teacher_overlay_audit_parser_uses_suffix_gate_defaults() -> None:
    args = overlay_audit.build_arg_parser().parse_args(
        ["--data", "track2p-root", "--output", "overlay.csv"]
    )

    assert args.threshold_method == "min"
    assert args.iou_distance_threshold == 12.0
    assert args.suffix_path_length == 2
    assert args.max_stitches_per_subject == 1


@pytest.mark.parametrize(
    "option",
    ["--suffix-path-length", "--max-stitches-per-subject", "--edge-top-k", "--path-beam-width"],
)
def test_coherence_teacher_overlay_parser_rejects_nonpositive_search_budgets(
    option: str,
) -> None:
    with pytest.raises(SystemExit):
        overlay_audit.build_arg_parser().parse_args(
            [
                "--data",
                "track2p-root",
                "--reference",
                "manual-gt",
                "--output",
                "overlay.csv",
                option,
                "0",
            ]
        )


def test_overlay_audit_fieldnames_match_requested_columns() -> None:
    assert overlay_audit.FIELDNAMES == (
        "subject",
        "session_a",
        "session_b",
        "roi_a",
        "roi_b",
        "already_in_component_cleanup",
        "already_in_coherence_suffix_stitch",
        "track2p_supported",
        "edge_status_against_gt",
        "creates_duplicate_source",
        "creates_duplicate_target",
        "would_break_complete_tp",
        "would_create_complete_fp",
        "pairwise_tp_delta_if_added",
        "pairwise_fp_delta_if_added",
        "pairwise_fn_delta_if_added",
        "complete_tp_delta_if_added",
        "complete_fp_delta_if_added",
        "complete_fn_delta_if_added",
    )


def test_conflict_flags_detect_existing_source_and_target_claims() -> None:
    source_matrix = np.array(
        [
            [10, 20, -1],
            [-1, 30, 40],
        ],
        dtype=int,
    )
    target_matrix = np.array(
        [
            [10, -1, -1],
            [12, 30, 40],
        ],
        dtype=int,
    )

    source_conflict = overlay_audit._conflict_flags(source_matrix, (0, 1, 10, 30))
    target_conflict = overlay_audit._conflict_flags(target_matrix, (0, 1, 11, 30))

    assert source_conflict == {"source": True, "target": False}
    assert target_conflict == {"source": False, "target": True}
