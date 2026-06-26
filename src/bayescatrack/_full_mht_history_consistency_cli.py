"""Install CLI registration for FullMHT history-consistency experiments."""

from __future__ import annotations

from types import ModuleType

_CANONICAL_BENCHMARK = "track2p-policy-full-mht-history-consistency"
_MODULE = (
    "bayescatrack.experiments."
    "track2p_policy_full_mht_history_consistency_benchmark"
)
_HELP = "Run FullMHT with label-free identity-history consistency scoring"
_ALIASES = (
    "track2p-full-mht-history-consistency",
    "track2p-pyrecest-full-mht-history-consistency",
    "track2p-component-full-mht-history-consistency",
)


def install_full_mht_history_consistency_cli(cli_module: ModuleType) -> None:
    """Expose the history-consistency FullMHT row in ``bayescatrack benchmark``."""

    benchmark_command = cli_module._BenchmarkCommand  # pylint: disable=protected-access
    cli_module._BENCHMARK_COMMANDS.setdefault(  # pylint: disable=protected-access
        _CANONICAL_BENCHMARK,
        benchmark_command(_MODULE, _HELP),
    )
    cli_module._BENCHMARK_ALIASES.update(  # pylint: disable=protected-access
        {alias: _CANONICAL_BENCHMARK for alias in _ALIASES}
    )


__all__ = ("install_full_mht_history_consistency_cli",)
