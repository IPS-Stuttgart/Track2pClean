"""Experiment runners and benchmark CLIs for BayesCaTrack."""

from . import _triplet_support_benchmark_integration as _triplet_support_benchmark_integration

_triplet_support_benchmark_integration.install_track2p_benchmark_triplet_support_integration()

__all__: list[str] = []