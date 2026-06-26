from __future__ import annotations

from bayescatrack import cli

# pylint: disable=protected-access


def test_full_mht_history_consistency_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-full-mht-history-consistency"]

    assert canonical == "track2p-policy-full-mht-history-consistency"
    assert (
        cli._BENCHMARK_ALIASES["track2p-pyrecest-full-mht-history-consistency"]
        == canonical
    )
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-full-mht-history-consistency"]
        == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments."
        "track2p_policy_full_mht_history_consistency_benchmark"
    )
