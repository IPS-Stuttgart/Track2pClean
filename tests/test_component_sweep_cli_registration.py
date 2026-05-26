from __future__ import annotations

from bayescatrack import cli

# pylint: disable=protected-access


def test_component_cleanup_sweep_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-component-cleanup-sweep"]

    assert canonical == "track2p-policy-component-sweep"
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_component_sweep"
    )


def test_multisplit_cleanup_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-multisplit-cleanup"]

    assert canonical == "track2p-policy-multisplit-cleanup"
    assert cli._BENCHMARK_ALIASES["track2p-component-multisplit-cleanup"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_multisplit_cleanup"
    )


def test_consensus_cleanup_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-consensus-cleanup"]

    assert canonical == "track2p-policy-consensus-cleanup"
    assert cli._BENCHMARK_ALIASES["track2p-component-consensus-cleanup"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_consensus_cleanup"
    )


def test_gap_pruned_policy_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-gap-pruned"]

    assert canonical == "track2p-policy-gap-pruned"
    assert cli._BENCHMARK_ALIASES["track2p-gap-rescue-pruned"] == canonical
    assert cli._BENCHMARK_ALIASES["track2p-policy-gap-rescue-pruned"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_gap_pruned_benchmark"
    )


def test_gap_component_cleanup_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-gap-component-cleanup"]

    assert canonical == "track2p-policy-gap-component-cleanup"
    assert cli._BENCHMARK_ALIASES["track2p-gap-rescue-component-cleanup"] == canonical
    assert cli._BENCHMARK_ALIASES["track2p-policy-gap-rescue-component-cleanup"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_gap_component_cleanup"
    )


def test_gap_consensus_cleanup_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-gap-consensus-cleanup"]

    assert canonical == "track2p-policy-gap-consensus-cleanup"
    assert cli._BENCHMARK_ALIASES["track2p-gap-rescue-consensus-cleanup"] == canonical
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-gap-consensus-cleanup"] == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_gap_consensus_cleanup"
    )
