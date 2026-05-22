"""Experiment runners and benchmark CLIs for BayesCaTrack."""

from . import _activity_sweep_defaults as _activity_sweep_defaults
from . import _cost_sweep_defaults as _cost_sweep_defaults
from . import (
    _triplet_support_benchmark_integration as _triplet_support_benchmark_integration,
)
from ._calibration_feature_registry_integration import (
    install_calibration_feature_registry_integration,
)

_triplet_support_benchmark_integration.install_track2p_benchmark_triplet_support_integration()
install_calibration_feature_registry_integration()
_cost_sweep_defaults.install_cost_sweep_suite2p_defaults()
_activity_sweep_defaults.install_activity_sweep_suite2p_defaults()

__all__: list[str] = []
