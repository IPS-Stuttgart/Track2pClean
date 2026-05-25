from __future__ import annotations

from bayescatrack import cli

# pylint: disable=protected-access


def test_component_cleanup_sweep_is_registered():
    canonical = cli._BENCHMARK_ALIASES["track2p-component-cleanup-sweep"]

    assert canonical == "track2p-policy-component-sweep"
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_component_sweep"
    )
