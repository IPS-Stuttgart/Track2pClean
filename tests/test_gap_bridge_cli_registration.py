from __future__ import annotations

from bayescatrack import cli

# pylint: disable=protected-access


def test_gap_bridge_cleanup_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-gap-bridge-cleanup"]

    assert canonical == "track2p-policy-gap-bridge-cleanup"
    assert cli._BENCHMARK_ALIASES["track2p-gap-rescue-bridge-cleanup"] == canonical
    assert (
        cli._BENCHMARK_ALIASES["track2p-policy-gap-rescue-bridge-cleanup"]
        == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_gap_bridge_cleanup"
    )
