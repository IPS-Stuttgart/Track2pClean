"""Install CLI registration for confidence-ordered strict gap cleanup."""

from __future__ import annotations

from types import ModuleType

_CANONICAL_BENCHMARK = "track2p-policy-confidence-ordered-strict-gated-gap-cleanup"
_MODULE = (
    "bayescatrack.experiments.track2p_policy_confidence_ordered_strict_gap_cleanup"
)
_HELP = "Run component cleanup plus confidence-ordered strictly gated gap rescue"
_ALIASES = (
    "track2p-confidence-ordered-strict-gap-cleanup",
    "track2p-confidence-ordered-strict-gated-gap-cleanup",
)


def install_confidence_ordered_strict_gap_cli(cli_module: ModuleType) -> None:
    """Expose the confidence-ordered strict-gap runner in ``bayescatrack benchmark``."""

    benchmark_command = cli_module._BenchmarkCommand  # pylint: disable=protected-access
    cli_module._BENCHMARK_COMMANDS.setdefault(  # pylint: disable=protected-access
        _CANONICAL_BENCHMARK,
        benchmark_command(_MODULE, _HELP),
    )
    cli_module._BENCHMARK_ALIASES.update(  # pylint: disable=protected-access
        {alias: _CANONICAL_BENCHMARK for alias in _ALIASES}
    )


__all__ = ("install_confidence_ordered_strict_gap_cli",)
