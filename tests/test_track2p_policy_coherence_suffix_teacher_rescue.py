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
