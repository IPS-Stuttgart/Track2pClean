from __future__ import annotations

from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_coherence_suffix_teacher_rescue


def test_coherence_suffix_teacher_rescue_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-coherence-suffix-teacher-rescue"]

    assert canonical == "track2p-policy-coherence-suffix-teacher-rescue"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-coherence-suffix-teacher-rescue"]
        == canonical
    )
    assert (
        cli._BENCHMARK_COMMANDS[canonical].module
        == "bayescatrack.experiments.track2p_policy_coherence_suffix_teacher_rescue"
    )


def test_coherence_suffix_teacher_rescue_defaults_match_manifest_teacher_row() -> None:
    args = track2p_policy_coherence_suffix_teacher_rescue.build_arg_parser().parse_args(
        ["--data", "track2p-root", "--output", "scores.csv"]
    )

    assert args.teacher_edge_order == "structural"
    assert args.teacher_action_filter == "all"
    assert args.teacher_feature_preset == "none"
    assert args.max_applied_teacher_edits == -1
    assert args.allow_completing_rescue is None
    assert args.allow_teacher_supported_completing_rescue is False
    assert args.allow_teacher_confirmed_completing_rescue is False


def test_coherence_suffix_teacher_rescue_exposes_action_specific_gates() -> None:
    parser = track2p_policy_coherence_suffix_teacher_rescue.build_arg_parser()
    args = parser.parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "scores.csv",
            "--teacher-edge-order",
            "dynamic-seed-cell-confidence",
            "--teacher-action-filter",
            "target-extension-or-seed-source-backfill",
            "--target-extension-feature-preset",
            "moderate-iou-cell-confidence",
            "--seed-source-feature-preset",
            "seed-source-cell-confident",
            "--no-allow-source-backfill",
            "--allow-seed-source-backfill",
            "--allow-completing-seed-source-backfill",
            "--no-allow-fragment-merges",
            "--min-teacher-component-observations",
            "2",
        ]
    )

    assert args.teacher_edge_order == "dynamic-seed-cell-confidence"
    assert args.teacher_action_filter == "target-extension-or-seed-source-backfill"
    assert args.target_extension_feature_preset == "moderate-iou-cell-confidence"
    assert args.seed_source_feature_preset == "seed-source-cell-confident"
    assert args.allow_source_backfill is False
    assert args.allow_seed_source_backfill is True
    assert args.allow_completing_seed_source_backfill is True
    assert args.allow_fragment_merges is False
    assert args.min_teacher_component_observations == 2


def test_coherence_suffix_teacher_rescue_exposes_completing_seed_source_filter() -> None:
    parser = track2p_policy_coherence_suffix_teacher_rescue.build_arg_parser()
    args = parser.parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "scores.csv",
            "--teacher-edge-order",
            "dynamic-seed-cell-confidence",
            "--teacher-action-filter",
            "completing-seed-source-backfill",
            "--seed-source-feature-preset",
            "seed-source-cell-confident",
            "--no-allow-source-backfill",
            "--allow-seed-source-backfill",
            "--allow-completing-seed-source-backfill",
            "--no-allow-fragment-merges",
            "--max-applied-teacher-edits",
            "1",
        ]
    )

    assert args.teacher_edge_order == "dynamic-seed-cell-confidence"
    assert args.teacher_action_filter == "completing-seed-source-backfill"
    assert args.seed_source_feature_preset == "seed-source-cell-confident"
    assert args.allow_source_backfill is False
    assert args.allow_seed_source_backfill is True
    assert args.allow_completing_seed_source_backfill is True
    assert args.allow_fragment_merges is False
    assert args.max_applied_teacher_edits == 1


def test_completing_filter_enables_completing_rescue_by_default() -> None:
    resolve = (
        track2p_policy_coherence_suffix_teacher_rescue._resolve_allow_completing_rescue
    )

    assert resolve(None, "completing-rescue") is True
    assert resolve(None, "completing_rescue") is True
    assert resolve(None, "completing-seed-source-backfill") is False
    assert resolve(None, "completing_seed_source_backfill") is False
    assert resolve(None, "all") is False
    assert resolve(False, "completing-rescue") is False
    assert resolve(True, "all") is True


def test_coherence_suffix_teacher_rescue_exposes_completing_rescue_override() -> None:
    parser = track2p_policy_coherence_suffix_teacher_rescue.build_arg_parser()

    auto_args = parser.parse_args(["--data", "track2p-root", "--output", "scores.csv"])
    off_args = parser.parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "scores.csv",
            "--no-allow-completing-rescue",
        ]
    )
    on_args = parser.parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "scores.csv",
            "--allow-completing-rescue",
        ]
    )

    assert auto_args.allow_completing_rescue is None
    assert off_args.allow_completing_rescue is False
    assert on_args.allow_completing_rescue is True


def test_coherence_suffix_teacher_rescue_exposes_teacher_supported_completion() -> None:
    parser = track2p_policy_coherence_suffix_teacher_rescue.build_arg_parser()

    args = parser.parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "scores.csv",
            "--no-allow-completing-rescue",
            "--allow-teacher-supported-completing-rescue",
            "--teacher-action-filter",
            "completing-rescue",
        ]
    )

    assert args.allow_completing_rescue is False
    assert args.allow_teacher_supported_completing_rescue is True


def test_coherence_suffix_teacher_rescue_exposes_teacher_confirmed_completion() -> None:
    parser = track2p_policy_coherence_suffix_teacher_rescue.build_arg_parser()

    args = parser.parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "scores.csv",
            "--no-allow-completing-rescue",
            "--allow-teacher-confirmed-completing-rescue",
            "--teacher-action-filter",
            "completing-rescue",
        ]
    )

    assert args.allow_completing_rescue is False
    assert args.allow_teacher_confirmed_completing_rescue is True
