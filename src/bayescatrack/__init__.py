"""Track2pClean public package API.

The implementation currently keeps the historical ``bayescatrack`` Python
namespace for backward compatibility. New user-facing documentation and the
console entry point use the Track2pClean name.
"""

# pylint: disable=duplicate-code

from . import _strict_config_validation as _strict_config_validation
from . import cli as _cli
from . import matching as _matching
from ._advanced_weight_validation import (
    install_advanced_weight_validation as _install_advanced_weight_validation,
)
from ._assignment_bundle_validation import (
    install_assignment_bundle_validation as _install_assignment_bundle_validation,
)
from ._bundle_session_roi_consistency_validation import (
    install_bundle_session_roi_consistency_validation as _install_bundle_session_roi_consistency_validation,
)
from ._calibrated_session_gap_validation import (
    install_calibrated_session_gap_validation as _install_calibrated_session_gap_validation,
)
from ._candidate_centroid_validation import (
    install_candidate_centroid_validation as _install_candidate_centroid_validation,
)
from ._cli_exit_code_validation import (
    install_cli_exit_code_validation as _install_cli_exit_code_validation,
)
from ._confidence_ordered_strict_gap_cli import (
    install_confidence_ordered_strict_gap_cli as _install_confidence_ordered_strict_gap_cli,
)
from ._edge_ranking_label_validation import (
    install_edge_ranking_label_validation as _install_edge_ranking_label_validation,
)
from ._empty_candidate_margin import (
    install_empty_candidate_gate_margin_fix as _install_empty_candidate_gate_margin_fix,
)
from ._empty_registered_roi_mask_validation import (
    install_empty_registered_roi_mask_validation as _install_empty_registered_roi_mask_validation,
)
from ._fov_affine_estimator_validation import (
    install_fov_affine_estimator_validation as _install_fov_affine_estimator_validation,
)
from ._fov_affine_validation import (
    install_fov_affine_warp_validation as _install_fov_affine_warp_validation,
)
from ._fov_subpixel_shift_validation import (
    install_fov_subpixel_shift_validation as _install_fov_subpixel_shift_validation,
)
from ._fov_translation_output_shape_validation import (
    install_fov_translation_output_shape_validation as _install_fov_translation_output_shape_validation,
)
from ._global_cost_preset_validation import (
    install_global_cost_preset_validation as _install_global_cost_preset_validation,
)
from ._global_track_row_validation import (
    install_global_track_row_validation as _install_global_track_row_validation,
)
from ._ground_truth_track_validation import (
    install_ground_truth_track_validation as _install_ground_truth_track_validation,
)
from ._growth_optional_roi_validation import (
    install_growth_optional_roi_validation as _install_growth_optional_roi_validation,
)
from ._growth_session_index_validation import (
    install_growth_session_index_validation as _install_growth_session_index_validation,
)
from ._integer_translation_validation import (
    install_integer_image_translation_validation as _install_integer_translation_validation,
)
from ._matching_bundle_roi_index_validation import (
    install_matching_bundle_roi_index_validation as _install_matching_bundle_roi_index_validation,
)
from ._matching_control_validation import (
    install_matching_control_validation as _install_matching_control_validation,
)
from ._matching_max_cost_validation import (
    install_matching_max_cost_validation as _install_matching_max_cost_validation,
)
from ._matching_validation import (
    install_matching_layout_validation as _install_matching_layout_validation,
)
from ._multisession_config_validation import (
    install_multisession_config_validation as _install_multisession_config_validation,
)
from ._multisession_solver_signature_validation import (
    install_multisession_solver_signature_validation as _install_multisession_solver_signature_validation,
)
from ._multisession_solver_track_validation import (
    install_multisession_solver_track_validation as _install_multisession_solver_track_validation,
)
from ._multisession_solver_validation import (
    install_multisession_solver_validation as _install_multisession_solver_validation,
)
from ._nonrigid_fov_image_validation import (
    install_nonrigid_fov_image_validation as _install_nonrigid_fov_image_validation,
)
from ._nonrigid_registration_control_validation import (
    install_nonrigid_registration_control_validation as _install_nonrigid_registration_control_validation,
)
from ._pairwise_return_components_validation import (
    install_return_components_validation as _install_return_components_validation,
)
from ._pyrecest_shifted_validation import (
    install_pyrecest_shifted_validation as _install_pyrecest_shifted_validation,
)
from ._reference_validation import (
    install_reference_validation as _install_reference_validation,
)
from ._registration_selection_validation import (
    install_registration_selection_validation as _install_registration_selection_validation,
)
from ._registration_warp_validation import (
    install_registration_warp_validation as _install_registration_warp_validation,
)
from ._roi_cue_length_validation import (
    install_roi_cue_length_validation as _install_roi_cue_length_validation,
)
from ._session_gap_validation import (
    install_session_gap_validation as _install_session_gap_validation,
)
from ._session_match_validation import (
    install_session_match_result_validation as _install_session_match_result_validation,
)
from ._shared_session_roi_validation import (
    install_shared_session_roi_validation as _install_shared_session_roi_validation,
)
from ._shifted_overlap_validation import (
    install_shifted_overlap_scalar_validation as _install_shifted_overlap_scalar_validation,
)
from ._soft_overlap_numeric_validation import (
    install_soft_overlap_numeric_validation as _install_soft_overlap_numeric_validation,
)
from ._strict_config_validation import (
    install_strict_config_validation as _install_strict_config_validation,
)
from ._suite2p_trace_bool_validation import (
    install_suite2p_trace_bool_validation as _install_suite2p_trace_bool_validation,
)
from ._suite2p_validation import (
    install_suite2p_stat_validation as _install_suite2p_stat_validation,
)
from ._supervised_mask_validation import (
    install_supervised_mask_validation as _install_supervised_mask_validation,
)
from ._track2p_policy_session_gap_validation import (
    install_track2p_policy_session_gap_validation as _install_track2p_policy_session_gap_validation,
)
from ._track_row_export_option_validation import (
    install_track_row_export_option_validation as _install_track_row_export_option_validation,
)
from ._track_row_fill_value_validation import (
    install_track_row_fill_value_validation as _install_track_row_fill_value_validation,
)
from ._track_table_session_name_validation import (
    install_track_table_session_name_validation as _install_track_table_session_name_validation,
)
from ._tracking_duplicate_start_roi_validation import (
    install_tracking_duplicate_start_roi_validation as _install_tracking_duplicate_start_roi_validation,
)
from ._tracking_fill_value_validation import (
    install_tracking_fill_value_validation as _install_tracking_fill_value_validation,
)
from ._tracking_global_link_edge_validation import (
    install_tracking_global_link_edge_validation as _install_tracking_global_link_edge_validation,
)
from ._tracking_result_matrix_validation import (
    install_tracking_result_matrix_validation as _install_tracking_result_matrix_validation,
)
from ._tracking_start_roi_availability_validation import (
    install_tracking_start_roi_availability_validation as _install_tracking_start_roi_availability_validation,
)
from ._tracking_start_roi_validation import (
    install_tracking_start_roi_validation as _install_tracking_start_roi_validation,
)
from .advanced_roi_components import (
    install_advanced_roi_components as _install_advanced_roi_components,
)
from .association import absence_model as _absence_model
from .association import calibrated_costs as _calibrated_costs
from .association import candidate_prefilter as _candidate_prefilter
from .association import track2p_policy_priors as _track2p_policy_priors
from .core import bridge as _bridge
from .evaluation import edge_ranking as _edge_ranking
from .soft_overlap_costs import (
    install_soft_overlap_costs as _install_soft_overlap_costs,
)

_install_suite2p_stat_validation(_bridge)
_install_suite2p_trace_bool_validation(_bridge)

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
_install_cli_exit_code_validation(_cli)
main = _cli.main
summarize_subject = _bridge.summarize_subject

_install_confidence_ordered_strict_gap_cli(_cli)
_install_matching_layout_validation(_matching)
_install_matching_max_cost_validation(_matching)
_install_shared_session_roi_validation(_matching)
_install_bundle_session_roi_consistency_validation(_matching)
_install_soft_overlap_costs()
_install_soft_overlap_numeric_validation()
_install_shifted_overlap_scalar_validation()
_install_pyrecest_shifted_validation()
_install_global_cost_preset_validation()
_install_candidate_centroid_validation(_candidate_prefilter)
_install_advanced_roi_components()
_install_advanced_weight_validation()
_install_assignment_bundle_validation()
_install_integer_translation_validation()
_install_reference_validation()
_install_edge_ranking_label_validation(_edge_ranking)
_install_fov_affine_estimator_validation()
_install_fov_affine_warp_validation()
_install_fov_translation_output_shape_validation()
_install_fov_subpixel_shift_validation()
_install_nonrigid_fov_image_validation()
_install_nonrigid_registration_control_validation()
_install_ground_truth_track_validation()
_install_growth_optional_roi_validation()
_install_growth_session_index_validation()
_install_registration_selection_validation()
_install_registration_warp_validation()
_install_strict_config_validation()
_strict_config_validation._positive_int = _candidate_prefilter._positive_int
_install_empty_candidate_gate_margin_fix()
_install_empty_registered_roi_mask_validation()
_install_matching_bundle_roi_index_validation()
_install_matching_control_validation()
_install_multisession_config_validation()
_install_multisession_solver_track_validation()
_install_multisession_solver_validation()
_install_multisession_solver_signature_validation()
_install_session_gap_validation()
_install_calibrated_session_gap_validation()
_install_session_match_result_validation()
_install_roi_cue_length_validation(_absence_model)
_install_supervised_mask_validation(_calibrated_costs)
_install_tracking_start_roi_validation()
_install_tracking_start_roi_availability_validation()
_install_tracking_duplicate_start_roi_validation()
_install_tracking_global_link_edge_validation()
_install_track_row_export_option_validation()
_install_track_row_fill_value_validation()
_install_track_table_session_name_validation()
_install_track2p_policy_session_gap_validation(_track2p_policy_priors)
_install_global_track_row_validation()
_install_tracking_fill_value_validation()
_install_tracking_result_matrix_validation()
_install_return_components_validation(_bridge)

__all__ = tuple(dict.fromkeys((*_bridge.__all__, "main")))
