from __future__ import annotations

import importlib
import sys
import types

import pytest

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
    assert (
        cli._BENCHMARK_ALIASES["track2p-policy-gap-rescue-component-cleanup"]
        == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_gap_component_cleanup"
    )


def test_strict_gated_gap_cleanup_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-strict-gated-gap-cleanup"]

    assert canonical == "track2p-policy-strict-gated-gap-cleanup"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-strict-gated-gap-cleanup"]
        == canonical
    )
    assert cli._BENCHMARK_ALIASES["track2p-component-strict-gap-cleanup"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_strict_gated_gap_cleanup"
    )


def test_gap_edge_audit_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-gap-edge-audit"]

    assert canonical == "track2p-policy-gap-edge-audit"
    assert cli._BENCHMARK_ALIASES["track2p-gap-rescue-edge-audit"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_gap_edge_audit"
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


def test_pyrecest_residual_mht_cleanup_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-pyrecest-residual-mht-cleanup"]

    assert canonical == "track2p-policy-pyrecest-residual-mht-cleanup"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-pyrecest-residual-mht-cleanup"]
        == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_pyrecest_residual_mht_cleanup"
    )


def test_pyrecest_frontier_mht_cleanup_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-pyrecest-frontier-mht-cleanup"]

    assert canonical == "track2p-policy-pyrecest-frontier-mht-cleanup"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-pyrecest-frontier-mht-cleanup"]
        == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_pyrecest_frontier_mht_cleanup"
    )


def test_pyrecest_safe_frontier_mht_cleanup_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-pyrecest-safe-frontier-mht-cleanup"]

    assert canonical == "track2p-policy-pyrecest-safe-frontier-mht-cleanup"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-pyrecest-safe-frontier-mht-cleanup"]
        == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_pyrecest_safe_frontier_mht_cleanup"
    )


def test_pyrecest_frontier_mht_defaults_use_verified_safe_cap(frontier_mht_module):
    args = frontier_mht_module._with_frontier_defaults([])

    cap_index = args.index("--max-veto-min-cell-probability")

    assert args[cap_index + 1] == "0.65"
    assert "0.80" not in args[cap_index + 1 : cap_index + 2]


def test_pyrecest_frontier_mht_user_equals_option_overrides_default(
    frontier_mht_module,
):
    args = frontier_mht_module._with_frontier_defaults(
        ["--max-veto-min-cell-probability=0.70"]
    )

    assert "--max-veto-min-cell-probability" not in args
    assert args[-1] == "--max-veto-min-cell-probability=0.70"


@pytest.fixture()
def frontier_mht_module(monkeypatch):
    tracking = types.ModuleType("pyrecest.tracking")
    tracking.ResidualEditCandidate = object
    tracking.ResidualMHTConfig = object
    tracking.enumerate_residual_hypotheses = lambda *args, **kwargs: ()
    tracking.select_residual_hypothesis = lambda *args, **kwargs: None

    pyrecest = types.ModuleType("pyrecest")
    pyrecest.tracking = tracking

    monkeypatch.setitem(sys.modules, "pyrecest", pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.tracking", tracking)
    module_names = (
        "bayescatrack.experiments.track2p_policy_pyrecest_residual_mht_cleanup",
        "bayescatrack.experiments.track2p_policy_pyrecest_frontier_mht_cleanup",
    )
    for module_name in module_names:
        sys.modules.pop(module_name, None)

    module = importlib.import_module(
        "bayescatrack.experiments.track2p_policy_pyrecest_frontier_mht_cleanup"
    )
    yield module

    for module_name in module_names:
        sys.modules.pop(module_name, None)
