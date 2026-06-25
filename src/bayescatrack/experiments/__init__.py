"""Experiment runners and benchmark CLIs for BayesCaTrack."""

from . import _activity_sweep_defaults as _activity_sweep_defaults
from . import _cost_sweep_defaults as _cost_sweep_defaults
from . import _diag_defaults as _diag_defaults
from . import _seed_session_validation as _seed_session_validation
from . import _teacher_rescue_manifest_integration as _teacher_rescue_manifest_integration
from . import (
    _triplet_support_benchmark_integration as _triplet_support_benchmark_integration,
)
from ._benchmark_roi_index_validation import (
    install_benchmark_roi_index_validation,
)
from ._calibration_feature_registry_integration import (
    install_calibration_feature_registry_integration,
)
from ._summary_output_format_integration import (
    install_summary_output_format_integration,
)
from ._teacher_rescue_edit_cap_manifest_integration import (
    install_teacher_rescue_edit_cap_manifest_integration,
)
from ._teacher_rescue_repair_preset_manifest_integration import (
    install_teacher_rescue_repair_preset_manifest_integration,
)

_triplet_support_benchmark_integration.install_track2p_benchmark_triplet_support_integration()
_seed_session_validation.install_seed_session_validation()
install_benchmark_roi_index_validation()
install_calibration_feature_registry_integration()
_teacher_rescue_manifest_integration.install_teacher_rescue_manifest_integration()
install_teacher_rescue_edit_cap_manifest_integration()
install_teacher_rescue_repair_preset_manifest_integration()
install_summary_output_format_integration()
_cost_sweep_defaults.install_cost_sweep_suite2p_defaults()
_activity_sweep_defaults.install_activity_sweep_suite2p_defaults()
_diag_defaults.install_diagnostic_suite2p_defaults()

__all__: list[str] = []
