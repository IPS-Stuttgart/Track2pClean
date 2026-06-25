"""BayesCaTrack public package API."""

# pylint: disable=duplicate-code

from . import cli as _cli
from ._advanced_weight_validation import (
    install_advanced_weight_validation as _install_advanced_weight_validation,
)
from ._assignment_bundle_validation import (
    install_assignment_bundle_validation as _install_assignment_bundle_validation,
)
from ._confidence_ordered_strict_gap_cli import (
    install_confidence_ordered_strict_gap_cli as _install_confidence_ordered_strict_gap_cli,
)
from ._fov_affine_validation import (
    install_fov_affine_warp_validation as _install_fov_affine_warp_validation,
)
from ._ground_truth_track_validation import (
    install_ground_truth_track_validation as _install_ground_truth_track_validation,
)
from ._integer_translation_validation import (
    install_integer_image_translation_validation as _install_integer_image_translation_validation,
)
from ._reference_validation import (
    install_reference_validation as _install_reference_validation,
)
from ._registration_selection_validation import (
    install_registration_selection_validation as _install_registration_selection_validation,
)
from ._session_gap_validation import (
    install_session_gap_validation as _install_session_gap_validation,
)
from ._session_match_validation import (
    install_session_match_result_validation as _install_session_match_result_validation,
)
from ._strict_config_validation import (
    install_strict_config_validation as _install_strict_config_validation,
)
from ._suite2p_validation import (
    install_suite2p_stat_validation as _install_suite2p_stat_validation,
)
from ._tracking_start_roi_validation import (
    install_tracking_start_roi_validation as _install_tracking_start_roi_validation,
)
from .advanced_roi_components import (
    install_advanced_roi_components as _install_advanced_roi_components,
)
from .core import bridge as _bridge
from .soft_overlap_costs import (
    install_soft_overlap_costs as _install_soft_overlap_costs,
)

_install_suite2p_stat_validation(_bridge)

CalciumPlaneData = _bridge.CalciumPlaneData
SessionAssociationBundle = _bridge.SessionAssociationBundle
Track2pSession = _bridge.Track2pSession
build_consecutive_session_association_bundles = (
    _bridge.build_consecutive_session_association_bundles
)
build_session_pair_association_bundle = _bridge.build_session_pair_association_bundle
export_subject_to_npz = _bridge.export_subject_to_npz
find_track2p_session_dirs = _bridge.find_track2p_session_dirs
load_raw_npy_plane = _bridge.load_raw_npy_plane
load_suite2p_plane = _bridge.load_suite2p_plane
load_track2p_subject = _bridge.load_track2p_subject
main = _cli.main
summarize_subject = _bridge.summarize_subject

_install_confidence_ordered_strict_gap_cli(_cli)
_install_soft_overlap_costs()
_install_advanced_roi_components()
_install_advanced_weight_validation()
_install_assignment_bundle_validation()
_install_integer_image_translation_validation()
_install_reference_validation()
_install_fov_affine_warp_validation()
_install_ground_truth_track_validation()
_install_registration_selection_validation()
_install_strict_config_validation()
_install_session_gap_validation()
_install_session_match_result_validation()
_install_tracking_start_roi_validation()

__all__ = tuple(dict.fromkeys((*_bridge.__all__, "main")))
